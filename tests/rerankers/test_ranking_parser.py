from ragkit.rerankers.ranking_parser import RankingParser


def test_parse_simple_ranking():

    parser = RankingParser()

    ranking = parser.parse(
        "2,1,3",
    )

    assert ranking == [1, 0, 2]


def test_parse_ignores_whitespace():

    parser = RankingParser()

    ranking = parser.parse(
        " 2 , 1 , 3 ",
    )

    assert ranking == [1, 0, 2]


def test_parse_empty_response():

    parser = RankingParser()

    ranking = parser.parse(
        "",
    )

    assert ranking == []


def test_parse_ignores_invalid_values():

    parser = RankingParser()

    ranking = parser.parse(
        "2,abc,1",
    )

    assert ranking == [1, 0]