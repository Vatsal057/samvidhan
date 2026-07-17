from samvidhan.chunking import chunk_pages

PAGES = [
    {
        "page": 3,
        "text": "19. Protection of freedom of speech. All citizens shall have the right "
        "to freedom of speech and expression. 20. Protection in respect of conviction. "
        "No person shall be convicted except for violation of a law in force.",
    },
    {
        "page": 4,
        "text": "21. Protection of life and personal liberty. No person shall be deprived "
        "of his life or personal liberty except according to procedure established by law.",
    },
]


def test_splits_on_article_boundaries():
    chunks = chunk_pages(PAGES)
    assert {c.article for c in chunks} == {"19", "20", "21"}


def test_page_metadata_follows_the_article():
    chunks = chunk_pages(PAGES)
    art21 = next(c for c in chunks if c.article == "21")
    assert art21.page == 4
    assert "personal liberty" in art21.text


def test_lettered_article_numbers_are_captured():
    pages = [{"page": 1, "text": "21A. Right to education. The State shall provide free "
             "and compulsory education to all children of the age of six to fourteen years."}]
    chunks = chunk_pages(pages)
    assert chunks[0].article == "21A"


def test_oversized_article_is_windowed_but_keeps_its_label():
    long_body = "21. Very long article. " + ("clause text " * 300)
    chunks = chunk_pages([{"page": 1, "text": long_body}], chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(c.article == "21" for c in chunks)


def test_text_without_headers_falls_back_gracefully():
    pages = [{"page": 1, "text": "This is arbitrary prose with no article numbers at all, "
             "long enough to survive the minimum-length fragment filter comfortably."}]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].article is None


def test_empty_input_returns_no_chunks():
    assert chunk_pages([]) == []
    assert chunk_pages([{"page": 1, "text": "   "}]) == []
