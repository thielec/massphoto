import pytest

import massphoto
from massphoto.defaults_store import DefaultsStore


def test_attribute_and_set_get_access():
    original = massphoto.defaults.psf_size
    massphoto.defaults.psf_size = 13
    assert massphoto.defaults.psf_size == 13

    massphoto.defaults.set("test.key", {"a": 1})
    assert massphoto.defaults.get("test.key") == {"a": 1}

    # Restore
    massphoto.defaults.psf_size = original

    # Delete test key
    value = massphoto.defaults.pop("test.key", None)
    assert value == {"a": 1}
    assert massphoto.defaults.get("test.key", default=None) is None


def test_get_set_state_roundtrip():
    state = DefaultsStore.get_state(copy=True)
    state["new.key"] = 123
    assert DefaultsStore.get("new.key", default=None) is None
    DefaultsStore.set_state(state, copy=True)
    assert DefaultsStore.get("new.key") == 123


def test_set_state_strips_meta_key():
    state = DefaultsStore.get_state(copy=True)
    state["new.key"] = 123
    state["__meta__"] = {"package_version": "0.0.0-test"}

    DefaultsStore.set_state(state, copy=True)

    assert DefaultsStore.get("new.key") == 123
    assert DefaultsStore.get("__meta__", default=None) is None


def test_set_state_warns_on_meta_version_mismatch(caplog):
    state = DefaultsStore.get_state(copy=True)
    state["__meta__"] = {"package_version": "0.0.0-test"}

    with caplog.at_level("WARNING"):
        DefaultsStore.set_state(state, copy=True)

    assert "does not match current package version" in caplog.text


def test_context_manager_restores_state():
    assert DefaultsStore.get("ctx.value") is None
    DefaultsStore.set("noctx.value", 2)
    with DefaultsStore.context({"ctx.value": 5}):
        assert DefaultsStore.get("ctx.value") == 5
        assert DefaultsStore.get("noctx.value") == 2
        # updating should only persist within context
        DefaultsStore.update({"ctx.value2": 10})
        assert DefaultsStore.get("ctx.value2") == 10
        DefaultsStore.set("ctx.value2", 11)
        # nested context
        with DefaultsStore.context({"ctx.value": 15}):
            assert DefaultsStore.get("ctx.value") == 15
        # exiting nested context should restore to outer context state
        assert DefaultsStore.get("ctx.value") == 5
        assert DefaultsStore.get("ctx.value2") == 11
    assert DefaultsStore.get("ctx.value") is None
    assert DefaultsStore.get("ctx.value2") is None
    assert DefaultsStore.get("noctx.value") == 2


def test_sync_and_apply_defaults_behaviour():
    @massphoto.defaults.sync_defaults(prefix="sync")
    def fn_sync(a=1, psf_size=11):
        return a, psf_size

    assert DefaultsStore.get("sync.psf_size") == 11
    assert fn_sync() == (1, 11)

    @massphoto.defaults.apply_defaults(prefix="apply")
    def fn_apply(a=1, psf_size=11):
        return a, psf_size

    massphoto.defaults.set("apply.psf_size", 15)
    assert fn_apply() == (1, 15)

    massphoto.defaults.set("fn_apply.psf_size", 17)
    assert fn_apply() == (1, 17)
    assert fn_apply(psf_size=19) == (1, 19)

    @massphoto.defaults.sync_apply_defaults(prefix="both")
    def fn_both(a=1, psf_size=11):
        return a, psf_size

    assert DefaultsStore.get("both.psf_size") == 11
    massphoto.defaults.set("both.psf_size", 21)
    assert fn_both() == (1, 21)


@massphoto.defaults.apply_defaults(prefix="pickle")
def fn_pickle(a=1, psf_size=11):
    return a, psf_size


def test_apply_defaults_with_pickle():
    import pickle

    data = pickle.dumps(fn_pickle)
    fn2 = pickle.loads(data)

    DefaultsStore.set("pickle.psf_size", 25)
    assert fn2() == (1, 25)


def test_wrap_callable_uses_snapshot_state():
    def fn():
        return DefaultsStore.get("wrap.value")

    DefaultsStore.set("wrap.value", 1)
    wrapped = DefaultsStore.wrap_callable(fn)
    DefaultsStore.set("wrap.value", 2)

    assert wrapped() == 1

    # double wrap: the inner snapshot should be overridden by the outer snapshot
    DefaultsStore.set("wrap.value", 3)
    double_wrapped = DefaultsStore.wrap_callable(wrapped)
    DefaultsStore.set("wrap.value", 4)
    assert double_wrapped() == 3


def test_save_and_load_roundtrip(tmp_path):
    DefaultsStore.set("io.value", 10)
    out_path = tmp_path / "defaults.yml"
    DefaultsStore.save(out_path)

    DefaultsStore.set("io.value", 20)
    DefaultsStore.load(out_path)

    assert DefaultsStore.get("io.value") == 10


@pytest.fixture(autouse=True)
def _restore_defaults_state():
    state = DefaultsStore.get_state(copy=True)
    yield
    DefaultsStore.set_state(state, copy=True)
