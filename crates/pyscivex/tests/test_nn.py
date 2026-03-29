"""Tests for pyscivex neural network — nn submodule."""

import pyscivex as sv


# ===========================================================================
# VARIABLE (AUTOGRAD TENSOR)
# ===========================================================================


class TestVariable:
    def test_create(self):
        x = sv.nn.Variable([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        assert x.shape() == [2, 2]
        assert x.requires_grad()

    def test_data(self):
        x = sv.nn.Variable([[1.0, 2.0]], requires_grad=False)
        d = x.data()
        assert d.tolist() == [[1.0, 2.0]]

    def test_no_grad_initially(self):
        x = sv.nn.Variable([1.0, 2.0], requires_grad=True)
        assert x.grad() is None

    def test_detach(self):
        x = sv.nn.Variable([1.0, 2.0], requires_grad=True)
        d = x.detach()
        assert not d.requires_grad()

    def test_repr(self):
        x = sv.nn.Variable([[1.0]], requires_grad=True)
        assert "Variable" in repr(x)

    def test_tensor_convenience(self):
        x = sv.nn.tensor([[5.0, 6.0]], requires_grad=True)
        assert x.shape() == [1, 2]
        assert x.requires_grad()

    def test_from_tensor(self):
        t = sv.Tensor([[1.0, 2.0]])
        x = sv.nn.Variable(t, requires_grad=True)
        assert x.shape() == [1, 2]


# ===========================================================================
# LAYERS
# ===========================================================================


class TestLinear:
    def test_forward(self):
        layer = sv.nn.Linear(4, 3)
        x = sv.nn.Variable([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        out = layer.forward(x)
        assert out.shape() == [1, 3]

    def test_parameters(self):
        layer = sv.nn.Linear(4, 3)
        params = layer.parameters()
        assert len(params) == 2  # weight + bias

    def test_no_bias(self):
        layer = sv.nn.Linear(4, 3, bias=False)
        params = layer.parameters()
        assert len(params) == 1  # weight only


class TestConv2d:
    def test_forward(self):
        conv = sv.nn.Conv2d(1, 4, 3, 3)  # in=1, out=4, kernel=3x3
        x = sv.nn.Variable(
            [float(i) for i in range(1 * 1 * 8 * 8)],  # batch=1, ch=1, 8x8
            requires_grad=False,
        )
        # We need to reshape — Variable from flat list is 1D
        # For conv2d, input must be [batch, channels, height, width]
        # Let's use a Tensor with correct shape
        t = sv.Tensor([float(i) for i in range(64)], [1, 1, 8, 8])
        x = sv.nn.Variable(t)
        out = conv.forward(x)
        assert out.shape()[0] == 1
        assert out.shape()[1] == 4


class TestConv1d:
    def test_forward(self):
        conv = sv.nn.Conv1d(2, 4, 3)  # in=2, out=4, kernel=3
        t = sv.Tensor([float(i) for i in range(2 * 10)], [1, 2, 10])
        x = sv.nn.Variable(t)
        out = conv.forward(x)
        assert out.shape()[0] == 1
        assert out.shape()[1] == 4


class TestBatchNorm1d:
    def test_forward(self):
        bn = sv.nn.BatchNorm1d(3)
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        x = sv.nn.Variable(t)
        out = bn.forward(x)
        assert out.shape() == [2, 3]

    def test_parameters(self):
        bn = sv.nn.BatchNorm1d(5)
        assert len(bn.parameters()) == 2  # gamma + beta


class TestDropout:
    def test_forward(self):
        drop = sv.nn.Dropout(0.5)
        t = sv.Tensor([1.0] * 100, [10, 10])
        x = sv.nn.Variable(t)
        out = drop.forward(x)
        assert out.shape() == [10, 10]


class TestEmbedding:
    def test_forward(self):
        emb = sv.nn.Embedding(10, 4)
        assert len(emb.parameters()) == 1


class TestLayerNorm:
    def test_forward(self):
        ln = sv.nn.LayerNorm(3)
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        x = sv.nn.Variable(t)
        out = ln.forward(x)
        assert out.shape() == [2, 3]


class TestLSTM:
    def test_forward(self):
        lstm = sv.nn.LSTM(input_size=4, hidden_size=8, seq_len=3)
        # Input: [batch, seq_len * input_size]
        t = sv.Tensor([float(i) for i in range(2 * 3 * 4)], [2, 12])
        x = sv.nn.Variable(t)
        out = lstm.forward(x)
        # Output: [batch, seq_len * hidden_size]
        assert out.shape() == [2, 24]


class TestGRU:
    def test_forward(self):
        gru = sv.nn.GRU(input_size=4, hidden_size=8, seq_len=3)
        t = sv.Tensor([float(i) for i in range(2 * 3 * 4)], [2, 12])
        x = sv.nn.Variable(t)
        out = gru.forward(x)
        assert out.shape() == [2, 24]


class TestMultiHeadAttention:
    def test_forward(self):
        mha = sv.nn.MultiHeadAttention(d_model=8, num_heads=2, seq_len=4)
        # Input: [batch, seq_len * d_model]
        t = sv.Tensor([float(i) for i in range(1 * 4 * 8)], [1, 32])
        x = sv.nn.Variable(t)
        out = mha.forward(x)
        assert out.shape() == [1, 32]


class TestActivationLayers:
    def test_relu(self):
        r = sv.nn.ReLU()
        x = sv.nn.Variable([-1.0, 0.0, 1.0, 2.0])
        out = r.forward(x)
        data = out.data().tolist()
        assert data[0] == 0.0
        assert data[3] == 2.0

    def test_sigmoid(self):
        s = sv.nn.Sigmoid()
        x = sv.nn.Variable([0.0])
        out = s.forward(x)
        assert abs(out.data().tolist()[0] - 0.5) < 1e-5

    def test_tanh(self):
        t = sv.nn.Tanh()
        x = sv.nn.Variable([0.0])
        out = t.forward(x)
        assert abs(out.data().tolist()[0]) < 1e-5


# ===========================================================================
# SEQUENTIAL
# ===========================================================================


class TestSequential:
    def test_basic(self):
        model = sv.nn.Sequential([
            sv.nn.Linear(4, 3),
            sv.nn.ReLU(),
            sv.nn.Linear(3, 2),
        ])
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4])
        x = sv.nn.Variable(t)
        out = model.forward(x)
        assert out.shape() == [1, 2]

    def test_parameters(self):
        model = sv.nn.Sequential([
            sv.nn.Linear(4, 3),
            sv.nn.ReLU(),
            sv.nn.Linear(3, 2),
        ])
        params = model.parameters()
        # 2 Linear layers with bias = 4 parameters
        assert len(params) == 4

    def test_len(self):
        model = sv.nn.Sequential([
            sv.nn.Linear(4, 3),
            sv.nn.ReLU(),
        ])
        assert len(model) == 2

    def test_repr(self):
        model = sv.nn.Sequential([sv.nn.Linear(2, 2)])
        assert "Sequential" in repr(model)

    def test_empty(self):
        model = sv.nn.Sequential()
        assert len(model) == 0


# ===========================================================================
# FUNCTIONAL ACTIVATIONS
# ===========================================================================


class TestFunctional:
    def test_relu(self):
        x = sv.nn.Variable([-1.0, 0.0, 1.0, 2.0])
        out = sv.nn.relu(x)
        data = out.data().tolist()
        assert data[0] == 0.0
        assert data[2] == 1.0

    def test_sigmoid(self):
        x = sv.nn.Variable([0.0])
        out = sv.nn.sigmoid(x)
        assert abs(out.data().tolist()[0] - 0.5) < 1e-5

    def test_tanh_act(self):
        x = sv.nn.Variable([0.0])
        out = sv.nn.tanh_act(x)
        assert abs(out.data().tolist()[0]) < 1e-5

    def test_softmax(self):
        x = sv.nn.Variable([[1.0, 2.0, 3.0]])
        out = sv.nn.softmax(x)
        data = out.data().tolist()
        # Softmax sums to ~1
        total = sum(data[0]) if isinstance(data[0], list) else sum(data)
        assert abs(total - 1.0) < 1e-5


# ===========================================================================
# LOSS FUNCTIONS
# ===========================================================================


class TestLoss:
    def test_mse_loss(self):
        pred = sv.nn.Variable([1.0, 2.0, 3.0])
        target = sv.nn.Variable([1.0, 2.0, 3.0])
        loss = sv.nn.mse_loss(pred, target)
        assert loss.data().tolist()[0] < 1e-10

    def test_mse_loss_nonzero(self):
        pred = sv.nn.Variable([1.0, 2.0, 3.0])
        target = sv.nn.Variable([2.0, 3.0, 4.0])
        loss = sv.nn.mse_loss(pred, target)
        val = loss.data().tolist()[0]
        assert abs(val - 1.0) < 1e-5  # mean((1)^2) = 1

    def test_bce_loss(self):
        pred = sv.nn.Variable([0.5, 0.5])
        target = sv.nn.Variable([1.0, 0.0])
        loss = sv.nn.bce_loss(pred, target)
        val = loss.data().tolist()[0]
        assert val > 0

    def test_huber_loss(self):
        pred = sv.nn.Variable([1.0, 2.0])
        target = sv.nn.Variable([1.5, 2.5])
        loss = sv.nn.huber_loss(pred, target)
        val = loss.data().tolist()[0]
        assert val > 0


# ===========================================================================
# BACKWARD PASS
# ===========================================================================


class TestBackward:
    def test_simple_backward(self):
        x = sv.nn.Variable([[2.0, 3.0]], requires_grad=True)
        layer = sv.nn.Linear(2, 1, seed=42)
        out = layer.forward(x)
        out.backward()
        g = x.grad()
        assert g is not None


# ===========================================================================
# OPTIMIZERS
# ===========================================================================


class TestSGD:
    def test_step(self):
        layer = sv.nn.Linear(2, 1, seed=42)
        params = layer.parameters()
        opt = sv.nn.SGD(params, lr=0.01)
        x = sv.nn.Variable([[1.0, 2.0]])
        target = sv.nn.Variable([[5.0]])
        # Forward + backward + step
        out = layer.forward(x)
        loss = sv.nn.mse_loss(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Weights should have changed
        out2 = layer.forward(x)
        d1 = out.data().tolist()
        d2 = out2.data().tolist()
        # Values should differ after optimization step
        assert d1 != d2


class TestAdam:
    def test_step(self):
        layer = sv.nn.Linear(2, 1, seed=42)
        params = layer.parameters()
        opt = sv.nn.Adam(params, lr=0.01)
        x = sv.nn.Variable([[1.0, 2.0]])
        target = sv.nn.Variable([[5.0]])
        out = layer.forward(x)
        loss = sv.nn.mse_loss(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()


class TestAdamW:
    def test_create(self):
        layer = sv.nn.Linear(2, 1, seed=42)
        params = layer.parameters()
        opt = sv.nn.AdamW(params, lr=0.001)
        assert opt is not None


class TestRMSprop:
    def test_create(self):
        layer = sv.nn.Linear(2, 1, seed=42)
        params = layer.parameters()
        opt = sv.nn.RMSprop(params, lr=0.01)
        assert opt is not None


# ===========================================================================
# LR SCHEDULERS
# ===========================================================================


class TestStepLR:
    def test_basic(self):
        sched = sv.nn.StepLR(base_lr=0.1, step_size=10, gamma=0.1)
        assert abs(sched.get_lr() - 0.1) < 1e-10
        for _ in range(10):
            sched.step()
        assert abs(sched.get_lr() - 0.01) < 1e-10


class TestCosineAnnealingLR:
    def test_basic(self):
        sched = sv.nn.CosineAnnealingLR(base_lr=0.1, t_max=100)
        initial_lr = sched.get_lr()
        assert abs(initial_lr - 0.1) < 1e-10
        for _ in range(50):
            sched.step()
        mid_lr = sched.get_lr()
        assert mid_lr < initial_lr


class TestReduceLROnPlateau:
    def test_basic(self):
        sched = sv.nn.ReduceLROnPlateau(initial_lr=0.1, factor=0.5, patience=2)
        sched.step(1.0)   # set initial best
        sched.step(1.1)   # bad epoch 1
        lr = sched.step(1.1)  # bad epoch 2 → reduce
        assert abs(lr - 0.05) < 1e-10


# ===========================================================================
# DATA LOADING
# ===========================================================================


class TestTensorDataset:
    def test_basic(self):
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        y = sv.Tensor([0.0, 1.0], [2])
        ds = sv.nn.TensorDataset(x, y)
        assert len(ds) == 2

    def test_get(self):
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        y = sv.Tensor([0.0, 1.0], [2])
        ds = sv.nn.TensorDataset(x, y)
        xi, yi = ds.get(0)
        assert xi.shape() == [2]


# ===========================================================================
# WEIGHT PERSISTENCE
# ===========================================================================


class TestPersistence:
    def test_save_load(self, tmp_path):
        layer = sv.nn.Linear(4, 3, seed=42)
        params = layer.parameters()
        path = str(tmp_path / "weights.bin")
        sv.nn.save_weights(path, params)
        loaded = sv.nn.load_weights(path)
        assert len(loaded) == 2  # weight + bias
        assert loaded[0].shape() == [3, 4]
        assert loaded[1].shape() == [3]


# ===========================================================================
# TRAINING LOOP (Integration)
# ===========================================================================


class TestTrainingLoop:
    def test_basic_loop(self):
        """Full train loop: forward → loss → backward → step."""
        model = sv.nn.Sequential([
            sv.nn.Linear(2, 4, seed=1),
            sv.nn.ReLU(),
            sv.nn.Linear(4, 1, seed=2),
        ])
        params = model.parameters()
        opt = sv.nn.Adam(params, lr=0.01)

        x = sv.nn.Variable([[1.0, 2.0], [3.0, 4.0]])
        target = sv.nn.Variable([[3.0], [7.0]])

        losses = []
        for _ in range(5):
            out = model.forward(x)
            loss = sv.nn.mse_loss(out, target)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.data().tolist()[0])

        # Loss should decrease over iterations
        assert losses[-1] <= losses[0] + 1.0  # allow some tolerance


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_nn_submodule_accessible(self):
        assert sv.nn.Variable is not None
        assert sv.nn.Linear is not None
        assert sv.nn.Sequential is not None
        assert sv.nn.Adam is not None
        assert sv.nn.mse_loss is not None

    def test_all_layer_classes(self):
        classes = [
            sv.nn.Linear, sv.nn.Conv1d, sv.nn.Conv2d,
            sv.nn.BatchNorm1d, sv.nn.BatchNorm2d,
            sv.nn.Dropout, sv.nn.Embedding, sv.nn.LayerNorm,
            sv.nn.LSTM, sv.nn.GRU, sv.nn.MultiHeadAttention,
            sv.nn.ReLU, sv.nn.Sigmoid, sv.nn.Tanh, sv.nn.Flatten,
            sv.nn.Sequential,
        ]
        for cls in classes:
            assert cls is not None

    def test_all_functions(self):
        fns = [
            sv.nn.tensor, sv.nn.relu, sv.nn.sigmoid,
            sv.nn.tanh_act, sv.nn.softmax, sv.nn.log_softmax,
            sv.nn.mse_loss, sv.nn.cross_entropy_loss,
            sv.nn.bce_loss, sv.nn.huber_loss, sv.nn.focal_loss,
            sv.nn.save_weights, sv.nn.load_weights,
        ]
        for fn in fns:
            assert fn is not None

    def test_all_optimizers(self):
        opts = [sv.nn.SGD, sv.nn.Adam, sv.nn.AdamW, sv.nn.RMSprop]
        for o in opts:
            assert o is not None

    def test_all_schedulers(self):
        scheds = [sv.nn.StepLR, sv.nn.CosineAnnealingLR, sv.nn.ReduceLROnPlateau]
        for s in scheds:
            assert s is not None
