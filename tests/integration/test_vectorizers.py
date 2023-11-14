import os

import pytest

from redisvl.vectorize.text import (
    HFTextVectorizer,
    OpenAITextVectorizer,
    VertexAITextVectorizer,
)


@pytest.fixture
def skip_vectorizer() -> bool:
    # os.getenv returns a string
    return os.getenv("SKIP_VECTORIZERS", "False").lower() == "true"


skip_vectorizer_test = lambda: pytest.config.getfixturevalue("skip_vectorizer")


@pytest.fixture(params=[HFTextVectorizer, OpenAITextVectorizer, VertexAITextVectorizer])
def vectorizer(request, openai_key, gcp_location, gcp_project_id):
    # if skip_vectorizer:
    #     pytest.skip("Skipping vectorizer tests")
    # Here we use actual models for integration test
    if request.param == HFTextVectorizer:
        return request.param(model="sentence-transformers/all-mpnet-base-v2")
    elif request.param == OpenAITextVectorizer:
        return request.param(
            model="text-embedding-ada-002", api_config={"api_key": openai_key}
        )
    elif request.param == VertexAITextVectorizer:
        # also need to set GOOGLE_APPLICATION_CREDENTIALS env var
        return request.param(
            model="textembedding-gecko",
            api_config={
                "location": gcp_location,
                "project_id": gcp_project_id,
            },
        )


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
def test_vectorizer_embed(vectorizer):
    text = "This is a test sentence."
    embedding = vectorizer.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
def test_vectorizer_embed_many(vectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = vectorizer.embed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == vectorizer.dims for emb in embeddings
    )


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
def test_vectorizer_bad_input(vectorizer):
    with pytest.raises(TypeError):
        vectorizer.embed(1)

    with pytest.raises(TypeError):
        vectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        vectorizer.embed_many(42)


@pytest.fixture(params=[OpenAITextVectorizer])
def avectorizer(request, openai_key):
    # Here we use actual models for integration test
    if request.param == OpenAITextVectorizer:
        return request.param(
            model="text-embedding-ada-002", api_config={"api_key": openai_key}
        )


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.asyncio
async def test_vectorizer_aembed(avectorizer):
    text = "This is a test sentence."
    embedding = await avectorizer.aembed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == avectorizer.dims


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.asyncio
async def test_vectorizer_aembed_many(avectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = await avectorizer.aembed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == avectorizer.dims for emb in embeddings
    )


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.asyncio
async def test_avectorizer_bad_input(avectorizer):
    with pytest.raises(TypeError):
        avectorizer.embed(1)

    with pytest.raises(TypeError):
        avectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        avectorizer.embed_many(42)
