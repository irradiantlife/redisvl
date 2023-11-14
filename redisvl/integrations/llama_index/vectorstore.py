"""Llama Index Redis Vector store index.

An index that that is built on top of an existing vector store.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import fsspec

from redisvl.utils.token_escaper import TokenEscaper
from redisvl.utils.utils import array_to_buffer, check_redis_modules_exist, convert_bytes
from redisvl.utils.connection import (
    check_connected,
    get_async_redis_connection,
    get_redis_connection,
)
from redisvl.index import (
    SearchIndex,
    AsyncSearchIndex
)
from redisvl.query import (
    VectorQuery,
    CountQuery,
    FilterQuery
)
from redisvl.query.filter import (
    Tag,
    Text,
    Num
)
from redisvl.schema import (
    TagFieldSchema,
    TextFieldSchema,
    NumericFieldSchema
)

from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.field import VectorField


class RedisVectorStore(VectorStore):
    stores_text = True
    stores_node = True
    flat_metadata = False

    tokenizer = TokenEscaper()

    DEFAULT_FIELDS = {
        "tag": [
            {"name": "id", "sortable": False},
            {"name": "doc_id", "sortable": False}
        ],
        "text": [
            {"name": "text", "weight": 1.0}
        ],
    }

    def __init__(
        self,
        index_name: str,
        index_prefix: str = "llama_index",
        prefix_ending: str = "/vector",
        index_args: Optional[Dict[str, Any]] = None,
        metadata_fields: Optional[List[str]] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RedisVectorStore.

        For index arguments that can be passed to RediSearch, see
        https://redis.io/docs/stack/search/reference/vectors/

        The index arguments will depend on the index type chosen. There
        are two available index types
            - FLAT: a flat index that uses brute force search
            - HNSW: a hierarchical navigable small world graph index

        Args:
            index_name (str): Name of the index.
            index_prefix (str): Prefix for the index. Defaults to "llama_index".
                The actual prefix used by Redis will be
                "{index_prefix}{prefix_ending}".
            prefix_ending (str): Prefix ending for the index. Be careful when
                changing this: https://github.com/jerryjliu/llama_index/pull/6665.
                Defaults to "/vector".
            index_args (Dict[str, Any]): Arguments for the index. Defaults to None.
            metadata_fields (List[str]): List of metadata fields to store in the index
                (only supports TAG fields).
            redis_url (str): URL for the redis instance.
                Defaults to "redis://localhost:6379".
            overwrite (bool): Whether to overwrite the index if it already exists.
                Defaults to False.
            kwargs (Any): Additional arguments to pass to the redis client.

        Raises:
            ValueError: If redis-py is not installed
            ValueError: If RediSearch is not installed

        Examples:
            >>> from llama_index.vector_stores.redis import RedisVectorStore
            >>> # Create a RedisVectorStore
            >>> vector_store = RedisVectorStore(
            >>>     index_name="my_index",
            >>>     index_prefix="llama_index",
            >>>     index_args={"algorithm": "HNSW", "m": 16, "ef_construction": 200,
                "distance_metric": "cosine"},
            >>>     redis_url="redis://localhost:6379/",
            >>>     overwrite=True)
        """
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        self._index_args = index_args if index_args is not None else {}
        self._metadata_fields = metadata_fields if metadata_fields is not None else []
        self._overwrite = overwrite
        self._vector_field = str(self._index_args.get("vector_field", "vector"))
        self._vector_key = str(self._index_args.get("vector_key", "vector"))


        # Create Index
        # add vector field to list of index fields. Create lazily to allow user
        # to specify index and search attributes in creation.

        fields = self.DEFAULT_FIELDS.copy()
        fields.update({"vector": [self._create_vector_field(self._vector_field, **self._index_args)]})

        # add metadata fields to list of index fields or we won't be able to search them
        for metadata_field in self._metadata_fields:
            # TODO: allow addition of text fields as metadata
            # TODO: make sure we're preventing overwriting other keys (e.g. text,
            #   doc_id, id, and other vector fields)
            fields["tags"].append({"name": metadata_field, "sortable": False})

        self.index = SearchIndex.from_dict({
            "index": {
                "name": index_name,
                "prefix": index_prefix+prefix_ending
            },
            "fields": self.DEFAULT_FIELDS
        })
        self.index.connect(redis_url, **kwargs)

    @property
    def client(self) -> "RedisType":
        """Return the redis client instance."""
        return self.index.client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.

        Raises:
            ValueError: If the index already exists and overwrite is False.
        """
        # check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        # set vector dim for creation if index doesn't exist
        self._index_args["dims"] = len(nodes[0].get_embedding())

        self.index.create(overwrite=self._overwrite)

        ids = []
        data = []
        for node in nodes:
            mapping = {
                "id": node.node_id,
                "doc_id": node.ref_doc_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                self._vector_key: array_to_buffer(node.get_embedding()),
            }
            additional_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            mapping.update(additional_metadata)

            ids.append(node.node_id)
            data.append(mapping)

        self.index.load(data, key_field="id", key_separator="_")
        # TODO need to handle this >>>
        # key = "_".join([self._prefix, str(node.node_id)])

        _logger.info(f"Added {len(ids)} documents to index {self.index._name}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # find all documents that match a doc_id
        doc_filter = Tag("doc_id") == ref_doc_id
        results = self.index.search(str(doc_filter))
        if len(results.docs) == 0:
            # don't raise an error but warn the user that document wasn't found
            # could be a result of eviction policy
            _logger.warning(
                f"Document with doc_id {ref_doc_id} not found "
                f"in index {self.index._name}"
            )
            return
        # clean up keys
        with self.index.client.pipeline(transaction=False) as pipe:
            for doc in results.docs:
                pipe.delete(doc.id)
            pipe.execute()

        _logger.info(
            f"Deleted {len(results.docs)} documents from index {self._index_name}"
        )

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        _logger.info(f"Deleting index {self.index._name}")
        self.index.delete()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        Raises:
            ValueError: If query.query_embedding is None.
            redis.exceptions.RedisError: If there is an error querying the index.
            redis.exceptions.TimeoutError: If there is a timeout querying the index.
            ValueError: If no documents are found when querying the index.
        """
        from redis.exceptions import RedisError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        return_fields = [
            "id",
            "doc_id",
            "text",
            self._vector_key,
            "vector_score",
            "_node_content",
        ]

        filters = _to_redis_filters(query.filters) if query.filters is not None else "*"

        _logger.info(f"Using filters: {filters}")

        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        redis_query = VectorQuery(
            vector=query.query_embedding,
            vector_field_name=self._vector_field,
            return_fields=return_fields,
            filter_expression=filters,
            num_results=query.similarity_top_k
        )
        _logger.info(f"Querying index {self.index._name}")

        try:
            results = self.index.query(redis_query)
        except RedisTimeoutError as e:
            _logger.error(f"Query timed out on {self.index._name}: {e}")
            raise
        except RedisError as e:
            _logger.error(f"Error querying {self.index._name}: {e}")
            raise

        if len(results) == 0:
            raise ValueError(
                f"No docs found on index '{self.index._name}' with "
                f"prefix '{self.index._prefix}' and filters '{filters}'. "
                "* Did you originally create the index with a different prefix? "
                "* Did you index your metadata fields when you created the index?"
            )

        ids = []
        nodes = []
        scores = []
        for doc in results:
            try:
                node = metadata_dict_to_node({"_node_content": doc["_node_content"]})
                node.text = doc["text"]
            except Exception:
                # TODO: Legacy support for old metadata format
                node = TextNode(
                    text=doc["text"],
                    id_=doc["id"],
                    embedding=None,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc["doc_id"])
                    },
                )
            ids.append(doc["id"])
            nodes.append(node)
            scores.append(1 - float(doc[redis_query.DISTANCE_ID]))

        _logger.info(f"Found {len(nodes)} results for query with id {ids}")
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def persist(
        self,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        in_background: bool = True,
    ) -> None:
        """Persist the vector store to disk.

        Args:
            persist_path (str): Path to persist the vector store to. (doesn't apply)
            in_background (bool, optional): Persist in background. Defaults to True.
            fs (fsspec.AbstractFileSystem, optional): Filesystem to persist to.
                (doesn't apply)

        Raises:
            redis.exceptions.RedisError: If there is an error
                                         persisting the index to disk.
        """
        from redis.exceptions import RedisError

        try:
            if in_background:
                _logger.info("Saving index to disk in background")
                self.index.client.bgsave()
            else:
                _logger.info("Saving index to disk")
                self.index.client.save()

        except RedisError as e:
            _logger.error(f"Error saving index to disk: {e}")
            raise

    def _create_vector_field(
        self,
        name: str,
        dims: int = 1536,
        algorithm: str = "FLAT",
        datatype: str = "FLOAT32",
        distance_metric: str = "COSINE",
        initial_cap: int = None,
        block_size: int = None,
        m: int = 16,
        ef_construction: int = 200,
        ef_runtime: int = 10,
        epsilon: float = 0.8,
        **kwargs: Any,
    ) -> "VectorField":
        """Create a RediSearch VectorField.

        Args:
            name (str): The name of the field.
            algorithm (str): The algorithm used to index the vector.
            dims (int): The dimensionality of the vector.
            datatype (str): The type of the vector. default: FLOAT32
            distance_metric (str): The distance metric used to compare vectors.
            initial_cap (int): The initial capacity of the index.
            block_size (int): The block size of the index.
            m (int): The number of outgoing edges in the HNSW graph.
            ef_construction (int): Number of maximum allowed potential outgoing edges
                            candidates for each node in the graph,
                            during the graph building.
            ef_runtime (int): The umber of maximum top candidates to hold during the
                KNN search

        Returns:
            A RediSearch VectorField.
        """
        if algorithm.upper() == "HNSW":
            return {
                "name": name,
                "type": "hnsw",
                "dims": dims,
                "distance": distance_metric.upper(),
                "m": m,
                "ef_construction": ef_construction,
                "ef_runtime": ef_runtime,
                "epsilon": epsilon,
                "initial_cap": initial_cap
            }
        else:
            return {
                "name": name,
                "type": "flat",
                "dims": dims,
                "distance": distance_metric.upper(),
                "initial_cap": initial_cap,
                "block_size": block_size,
            }



# currently only supports exact tag match - {} denotes a tag
# must create the index with the correct metadata field before using a field as a
#   filter, or it will return no results
def _to_redis_filters(metadata_filters: MetadataFilters) -> str:
    tokenizer = TokenEscaper()

    filter_strings = []
    for filter in metadata_filters.filters:
        # adds quotes around the value to ensure that the filter is treated as an
        #   exact match
        filter_string = f"@{filter.key}:{{{tokenizer.escape(str(filter.value))}}}"
        filter_strings.append(filter_string)

    joined_filter_strings = " & ".join(filter_strings)
    return f"({joined_filter_strings})"