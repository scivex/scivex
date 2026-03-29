"""Comprehensive tests for pyscivex DataFrame and I/O module."""

import pytest
import pyscivex as sv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_numeric_df():
    """3-row, 2-column float DataFrame."""
    return sv.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


def _make_mixed_df():
    """3-row DataFrame with float and string columns."""
    df = sv.DataFrame({"score": [10.0, 20.0, 30.0]})
    df.add_string_column("name", ["alice", "bob", "carol"])
    return df


def _make_groupby_df():
    """DataFrame suitable for groupby tests."""
    return sv.DataFrame({
        "group": [1.0, 1.0, 2.0, 2.0, 3.0],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


# ===================================================================
# TestDataFrameCreation
# ===================================================================

class TestDataFrameCreation:
    def test_empty(self):
        df = sv.DataFrame()
        assert df.nrows() == 0
        assert df.ncols() == 0
        assert df.is_empty()

    def test_from_dict_float(self):
        df = sv.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert df.nrows() == 2
        assert df.ncols() == 2
        assert df.column("a") == [1.0, 2.0]
        assert df.column("b") == [3.0, 4.0]

    def test_from_dict_string(self):
        df = sv.DataFrame({"name": ["x", "y", "z"]})
        assert df.nrows() == 3
        assert df.column_str("name") == ["x", "y", "z"]

    def test_from_dict_mixed(self):
        df = sv.DataFrame({"val": [1.0, 2.0], "label": ["a", "b"]})
        assert df.nrows() == 2
        assert df.ncols() == 2
        assert df.column("val") == [1.0, 2.0]
        assert df.column_str("label") == ["a", "b"]

    def test_from_dict_int(self):
        df = sv.DataFrame()
        df.add_int_column("ids", [10, 20, 30])
        assert df.nrows() == 3
        assert df.ncols() == 1


# ===================================================================
# TestDataFrameColumns
# ===================================================================

class TestDataFrameColumns:
    def test_add_column(self):
        df = sv.DataFrame()
        df.add_column("x", [1.0, 2.0, 3.0])
        assert df.ncols() == 1
        assert df.column("x") == [1.0, 2.0, 3.0]

    def test_add_string_column(self):
        df = sv.DataFrame()
        df.add_string_column("s", ["hello", "world"])
        assert df.ncols() == 1
        assert df.column_str("s") == ["hello", "world"]

    def test_add_int_column(self):
        df = sv.DataFrame()
        df.add_int_column("i", [7, 8, 9])
        assert df.ncols() == 1
        assert df.nrows() == 3

    def test_column_access(self):
        df = _make_numeric_df()
        assert df.column("y") == [4.0, 5.0, 6.0]

    def test_column_str(self):
        df = _make_mixed_df()
        assert df.column_str("name") == ["alice", "bob", "carol"]

    def test_getitem_str(self):
        df = _make_numeric_df()
        col = df["x"]
        assert col == [1.0, 2.0, 3.0]

    def test_getitem_list(self):
        df = _make_numeric_df()
        sub = df[["x", "y"]]
        assert sub.ncols() == 2
        assert sub.column("x") == [1.0, 2.0, 3.0]


# ===================================================================
# TestDataFrameShape
# ===================================================================

class TestDataFrameShape:
    def test_nrows(self):
        df = _make_numeric_df()
        assert df.nrows() == 3

    def test_ncols(self):
        df = _make_numeric_df()
        assert df.ncols() == 2

    def test_shape(self):
        df = _make_numeric_df()
        assert df.shape() == (3, 2)

    def test_column_names(self):
        df = _make_numeric_df()
        names = df.column_names()
        assert "x" in names
        assert "y" in names
        assert len(names) == 2

    def test_dtypes(self):
        df = _make_numeric_df()
        dtypes = df.dtypes()
        assert isinstance(dtypes, (list, dict))

    def test_is_empty_true(self):
        df = sv.DataFrame()
        assert df.is_empty()

    def test_is_empty_false(self):
        df = _make_numeric_df()
        assert not df.is_empty()

    def test_len(self):
        df = _make_numeric_df()
        assert len(df) == 3


# ===================================================================
# TestDataFrameSelection
# ===================================================================

class TestDataFrameSelection:
    def test_select(self):
        df = sv.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        sub = df.select(["a", "c"])
        assert sub.ncols() == 2
        assert sub.column_names() == ["a", "c"]

    def test_drop_columns(self):
        df = sv.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        dropped = df.drop_columns(["b"])
        assert "b" not in dropped.column_names()
        assert dropped.ncols() == 2

    def test_head(self):
        df = sv.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        h = df.head(3)
        assert h.nrows() == 3
        assert h.column("x") == [1.0, 2.0, 3.0]

    def test_tail(self):
        df = sv.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        t = df.tail(2)
        assert t.nrows() == 2
        assert t.column("x") == [4.0, 5.0]

    def test_slice(self):
        df = sv.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        s = df.slice(1, 3)
        assert s.nrows() == 3
        assert s.column("x") == [20.0, 30.0, 40.0]

    def test_filter(self):
        df = sv.DataFrame({"x": [1.0, 2.0, 3.0]})
        f = df.filter([True, False, True])
        assert f.nrows() == 2
        assert f.column("x") == [1.0, 3.0]


# ===================================================================
# TestDataFrameSorting
# ===================================================================

class TestDataFrameSorting:
    def test_sort_ascending(self):
        df = sv.DataFrame({"x": [3.0, 1.0, 2.0]})
        s = df.sort_values("x", ascending=True)
        assert s.column("x") == [1.0, 2.0, 3.0]

    def test_sort_descending(self):
        df = sv.DataFrame({"x": [3.0, 1.0, 2.0]})
        s = df.sort_values("x", ascending=False)
        assert s.column("x") == [3.0, 2.0, 1.0]


# ===================================================================
# TestDataFrameGroupBy
# ===================================================================

class TestDataFrameGroupBy:
    def test_sum(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).sum()
        assert result.nrows() == 3  # 3 distinct groups

    def test_mean(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).mean()
        assert result.nrows() == 3

    def test_min(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).min()
        assert result.nrows() == 3

    def test_max(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).max()
        assert result.nrows() == 3

    def test_count(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).count()
        assert result.nrows() == 3

    def test_first(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).first()
        assert result.nrows() == 3

    def test_last(self):
        df = _make_groupby_df()
        result = df.groupby(["group"]).last()
        assert result.nrows() == 3


# ===================================================================
# TestDataFrameJoin
# ===================================================================

class TestDataFrameJoin:
    def test_inner_join(self):
        left = sv.DataFrame({"key": [1.0, 2.0, 3.0], "lval": [10.0, 20.0, 30.0]})
        right = sv.DataFrame({"key": [2.0, 3.0, 4.0], "rval": [200.0, 300.0, 400.0]})
        joined = left.join(right, on=["key"], how="inner")
        assert joined.nrows() == 2  # keys 2 and 3 match

    def test_left_join(self):
        left = sv.DataFrame({"key": [1.0, 2.0, 3.0], "lval": [10.0, 20.0, 30.0]})
        right = sv.DataFrame({"key": [2.0, 3.0, 4.0], "rval": [200.0, 300.0, 400.0]})
        joined = left.join(right, on=["key"], how="left")
        assert joined.nrows() == 3  # all left rows preserved


# ===================================================================
# TestDataFramePivot
# ===================================================================

class TestDataFramePivot:
    def test_basic_pivot(self):
        df = sv.DataFrame({
            "row": [1.0, 1.0, 2.0, 2.0],
            "col": [1.0, 2.0, 1.0, 2.0],
            "val": [10.0, 20.0, 30.0, 40.0],
        })
        pivoted = df.pivot(index=["row"], columns="col", values="val", agg_func="sum")
        assert pivoted.nrows() == 2


# ===================================================================
# TestDataFrameMissing
# ===================================================================

class TestDataFrameMissing:
    def test_drop_nulls_no_nulls(self):
        df = _make_numeric_df()
        cleaned = df.drop_nulls()
        assert cleaned.nrows() == df.nrows()


# ===================================================================
# TestDataFrameSQL
# ===================================================================

class TestDataFrameSQL:
    def test_basic_select(self):
        df = sv.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = df.sql("SELECT * FROM t")
        assert result.nrows() == 3

    def test_where_clause(self):
        df = sv.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = df.sql("SELECT * FROM t WHERE x > 1")
        assert result.nrows() == 2


# ===================================================================
# TestDataFrameTensor
# ===================================================================

class TestDataFrameTensor:
    def test_to_tensor_basic(self):
        df = sv.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = df.to_tensor()
        # Should be a Tensor with shape (2, 2)
        assert t.shape() == [2, 2]


# ===================================================================
# TestDataFrameDisplay
# ===================================================================

class TestDataFrameDisplay:
    def test_repr(self):
        df = _make_numeric_df()
        r = repr(df)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_describe(self):
        df = _make_numeric_df()
        d = df.describe()
        assert isinstance(d, str)
        assert len(d) > 0


# ===================================================================
# TestIO — CSV, JSON, Parquet round-trips
# ===================================================================

class TestIO:
    def test_csv_roundtrip(self, tmp_path):
        df = _make_numeric_df()
        path = str(tmp_path / "test.csv")
        sv.io.write_csv(df, path)
        loaded = sv.io.read_csv(path)
        assert loaded.nrows() == 3
        assert loaded.ncols() == 2

    def test_csv_with_options(self, tmp_path):
        df = sv.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        path = str(tmp_path / "test_opts.csv")
        sv.io.write_csv(df, path, delimiter=";")
        loaded = sv.io.read_csv(path, delimiter=";", has_header=True)
        assert loaded.nrows() == 2

    def test_csv_no_header(self, tmp_path):
        df = sv.DataFrame({"a": [1.0, 2.0]})
        path = str(tmp_path / "noheader.csv")
        sv.io.write_csv(df, path, write_header=False)
        loaded = sv.io.read_csv(path, has_header=False)
        assert loaded.nrows() == 2

    def test_json_roundtrip(self, tmp_path):
        df = _make_numeric_df()
        path = str(tmp_path / "test.json")
        sv.io.write_json(df, path)
        loaded = sv.io.read_json(path)
        assert loaded.nrows() == 3
        assert loaded.ncols() == 2

    def test_json_pretty(self, tmp_path):
        df = sv.DataFrame({"v": [1.0, 2.0]})
        path = str(tmp_path / "pretty.json")
        sv.io.write_json(df, path, pretty=True)
        loaded = sv.io.read_json(path)
        assert loaded.nrows() == 2

    def test_csv_max_rows(self, tmp_path):
        df = sv.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        path = str(tmp_path / "big.csv")
        sv.io.write_csv(df, path)
        loaded = sv.io.read_csv(path, max_rows=3)
        assert loaded.nrows() == 3
