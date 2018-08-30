import torch.nn as nn
from torch.nn import functional
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import basic
from src.utils import *

# for debug
from IPython import embed


class AttnCombiner(nn.Module):
    def __init__(self, bidirectional, hidden_size, attention_size=128, dropout=0):
        super(AttnCombiner, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.ws1 = nn.Linear(hidden_size * self.num_directions, attention_size, bias=False)
        self.ws2 = nn.Linear(attention_size, 1, bias=False)
        init.orthogonal(self.ws1.weight.data)
        init.orthogonal(self.ws2.weight.data)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, hiddens):
        size = hiddens.size()  # [bsz, len, in_dim]
        x_flat = hiddens.contiguous().view(-1, size[2])  # [bsz*len, in_dim]
        h_bar = self.tanh(self.ws1(self.drop(x_flat)))  # [bsz*len, attn_hid]
        alphas = self.ws2(h_bar).view(size[0], size[1])  # [bsz, len]
        return alphas

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

    def forward(self, left=None, right=None):
        """
        Args:
            left: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            right: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = left
        hr, cr = right
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = basic.apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TreeLSTMEncoder(nn.Module):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional):
        super(TreeLSTMEncoder, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.pooling_method = pooling_method
        self.bidirectional = bidirectional

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim * 2 if self.bidirectional else hidden_dim)

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal(self.leaf_rnn_cell.weight_hh.data)
            init.constant(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal(self.word_linear.weight.data)
            init.constant(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2).expand_as(new_h)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def forward(self, *inp):
        pass


class GumbelTreeLSTMEncoder(TreeLSTMEncoder):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional, gumbel_temperature):
        super(GumbelTreeLSTMEncoder, self).__init__(
            word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional)
        self.gumbel_temperature = gumbel_temperature
        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim * 2 if self.bidirectional else hidden_dim))
        if pooling_method == 'attention':
            self.combiner = AttnCombiner(bidirectional, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        super(GumbelTreeLSTMEncoder, self).reset_parameters()
        init.normal(self.comp_query.data, mean=0, std=0.01)

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = basic.dot_nd(query=self.comp_query, candidates=new_h)
        if self.training:
            select_mask = basic.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature, mask=mask)
        else:
            select_mask = basic.greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, inp, length, return_select_masks=False):
        max_depth = inp.size(1)
        length_mask = basic.sequence_mask(sequence_length=length, max_length=max_depth)
        select_masks = list()
        features = list()

        if self.use_leaf_rnn:
            hs = list()
            cs = list()
            batch_size, max_length, _ = inp.size()
            zero_state = Variable(inp.data.new(batch_size, self.hidden_dim)
                                  .zero_())
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=inp[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = list()
                cs_bw = list()
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=inp, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = basic.apply_nd(fn=self.word_linear, input=inp)
            state = state.chunk(chunks=2, dim=2)
        nodes = list()
        if self.pooling_method is not None:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            left = (h[:, :-1, :], c[:, :-1, :])
            right = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(left=left, right=right)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i+1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                features.append(selected_h)
                if self.pooling_method is not None:
                    nodes.append(selected_h.unsqueeze(1))
            done_mask = length_mask[:, i+1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if (self.pooling_method is not None) and i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.pooling_method == 'max':
            nodes = torch.cat(nodes, dim=1)
            h = nodes.max(1)[0].unsqueeze(1)
        elif self.pooling_method == 'mean':
            nodes = torch.cat(nodes, dim=1).sum(1)
            lengths = length * 2 - 1
            lengths = lengths.unsqueeze(1).float().expand_as(nodes)
            h = (nodes / lengths).unsqueeze(1)
        elif self.pooling_method == 'attention':
            nodes = torch.cat(nodes, dim=1)
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask_expand = att_mask.float().unsqueeze(2).expand_as(nodes)
            att_weights = basic.masked_softmax(
                logits=self.combiner(nodes), mask=att_mask)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            h = (att_weights_expand * att_mask_expand * nodes).sum(1).unsqueeze(1)
        else:
            assert self.pooling_method is None
        assert h.size(1) == 1 and c.size(1) == 1
        if not return_select_masks:
            return h.squeeze(1), c.squeeze(1)
        else:
            return h.squeeze(1), c.squeeze(1), features, select_masks


class RecursiveTreeLSTMEncoder(TreeLSTMEncoder):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional):
        super(RecursiveTreeLSTMEncoder, self).__init__(
            word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional)
        self.reset_parameters()
        self.pooling_method = pooling_method
        if self.pooling_method == 'attention':
            self.combiner = AttnCombiner(bidirectional, hidden_dim)

    def forward(self, inp, length, fixed_masks):
        max_depth = inp.size(1)
        length_mask = basic.sequence_mask(sequence_length=length, max_length=max_depth)
        features = list()

        if self.use_leaf_rnn:
            hs = list()
            cs = list()
            batch_size, max_length, _ = inp.size()
            zero_state = Variable(inp.data.new(batch_size, self.hidden_dim).zero_())
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=inp[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = list()
                cs_bw = list()
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=inp, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = basic.apply_nd(fn=self.word_linear, input=inp)
            state = state.chunk(chunks=2, dim=2)
        nodes = list()
        if self.pooling_method is not None:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            left = (h[:, :-1, :], c[:, :-1, :])
            right = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(left=left, right=right)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, selected_h = self.compose(
                    old_state=state, new_state=new_state, tree_mask=fixed_masks[i])
                new_state = (new_h, new_c)
                features.append(selected_h)
                if self.pooling_method is not None:
                    nodes.append(selected_h.unsqueeze(1))
            done_mask = length_mask[:, i + 1]
            state = self.update_state(
                old_state=state, new_state=new_state, done_mask=done_mask)
            if (self.pooling_method is not None) and i >= max_depth - 2:
                done_masks = done_mask.float().unsqueeze(1).unsqueeze(2).expand_as(state[0])
                nodes.append(state[0] * done_masks)
        h, c = state
        if self.pooling_method == 'max':
            nodes = torch.cat(nodes, dim=1)
            h = nodes.max(1)[0].unsqueeze(1)
        elif self.pooling_method == 'mean':
            nodes = torch.cat(nodes, dim=1).sum(1)
            lengths = length * 2 - 1
            lengths = lengths.unsqueeze(1).float().expand_as(nodes)
            h = (nodes / lengths).unsqueeze(1)
        elif self.pooling_method == 'attention':
            nodes = torch.cat(nodes, dim=1)
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask_expand = att_mask.float().unsqueeze(2).expand_as(nodes)
            att_weights = basic.masked_softmax(
                logits=self.combiner(nodes), mask=att_mask)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            h = (att_weights_expand * att_mask_expand * nodes).sum(1).unsqueeze(1)
        else:
            assert self.pooling_method is None
        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1)

    @staticmethod
    def compose(old_state, new_state, tree_mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        select_mask_expand = tree_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = tree_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, selected_h

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LinearLSTMEncoder(nn.Module):
    def __init__(self, word_dim, hidden_dim, pooling_method, bidirectional):
        super(LinearLSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(word_dim, hidden_dim, 1, bidirectional=bidirectional)
        self.n_layers = 1
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_dim
        self.pooling_method = pooling_method
        if self.pooling_method == 'attention':
            self.combiner = AttnCombiner(bidirectional, hidden_dim)

    def reset_parameters(self):
        init.kaiming_normal(self.rnn.weight_ih_l0.data)
        init.orthogonal(self.rnn.weight_hh_l0.data)
        init.constant(self.rnn.bias_ih_l0.data, val=0)
        init.constant(self.rnn.bias_hh_l0.data, val=0)
        # Set forget bias to 1
        self.rnn.bias_ih_l0.data.chunk(4)[1].fill_(1)

    def forward(self, inp, length, h_0=None):
        inp, length, inv = sort_sentences_by_lengths(inp, length)
        if h_0 is None:
            h_size = (self.n_layers * self.num_directions, inp.size(0), self.hidden_size)
            h_0 = (
                Variable(inp.data.new(*h_size).zero_(), requires_grad=False),
                Variable(inp.data.new(*h_size).zero_(), requires_grad=False)
            )
        inp = pack_padded_sequence(inp, length.data.tolist(), batch_first=True)
        hiddens, h_n = self.rnn(inp, h_0)
        if self.pooling_method == 'attention':
            hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=0)
            att_weights = pack_padded_sequence(self.combiner(hiddens), length.data.tolist(), batch_first=True)
            att_weights, _ = pad_packed_sequence(att_weights, batch_first=True, padding_value=-1e8)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(hiddens)
            encodings = (att_weights_expand * hiddens).sum(1)
            return encodings.index_select(0, inv), None
        elif self.pooling_method == 'max':
            hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=-1e8)
            encodings = hiddens.max(1)[0].contiguous()
            return encodings.index_select(0, inv), None
        elif self.pooling_method == 'mean':
            hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=0)
            encodings = hiddens.sum(1)
            lengths_expand = length.float().unsqueeze(1).expand_as(encodings)
            encodings = encodings / lengths_expand
            return encodings.index_select(0, inv), None
        else:  # last
            assert self.pooling_method is None
            hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=-1e8)
            if self.num_directions == 1:
                forward_encodings = [hiddens[i, length[i].data[0] - 1, :].contiguous().view(1, -1)
                                     for i in range(hiddens.size(0))]
                encodings = torch.cat(forward_encodings, dim=0)
            else:
                forward_encodings = [hiddens[i, length[i].data[0] - 1, :self.hidden_size].contiguous().view(1, -1)
                                     for i in range(hiddens.size(0))]
                encodings = torch.cat((torch.cat(forward_encodings, dim=0),
                                       hiddens[:, 0, self.hidden_size:]), dim=1)
            return encodings.index_select(0, inv), None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Classifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers,
                 use_batchnorm, dropout_prob):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm

        if use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        mlp_layers = list()
        for i in range(num_layers):
            layer_in_features = hidden_dim if i > 0 else input_dim
            linear_layer = nn.Linear(in_features=layer_in_features,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            mlp_layer = nn.Sequential(linear_layer, relu_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        for i in range(self.num_layers):
            linear_layer = self.mlp[i][0]
            init.kaiming_normal(linear_layer.weight.data)
            init.constant(linear_layer.bias.data, val=0)
        init.uniform(self.clf_linear.weight.data, -0.005, 0.005)
        init.constant(self.clf_linear.bias.data, val=0)

    def forward(self, mlp_input):
        if self.use_batchnorm:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.use_batchnorm:
            mlp_output = self.bn_mlp_output(mlp_output)
        mlp_output = self.dropout(mlp_output)
        logits = self.clf_linear(mlp_output)
        return logits

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SentClassModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, pooling_method, clf_num_layers,
                 use_batchnorm, dropout_prob, bidirectional, encoder_type,
                 use_leaf_rnn=False):
        super(SentClassModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.pooling_method = pooling_method
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.encoder_type = encoder_type

        self.word_embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=word_dim)
        if encoder_type == 'gumbel':
            self.encoder = GumbelTreeLSTMEncoder(
                word_dim=word_dim, hidden_dim=hidden_dim,
                use_leaf_rnn=use_leaf_rnn,
                pooling_method=pooling_method,
                gumbel_temperature=1,
                bidirectional=bidirectional,
            )
        elif encoder_type in ['parsing', 'balanced', 'left', 'right']:
            self.encoder = RecursiveTreeLSTMEncoder(
                word_dim=word_dim, hidden_dim=hidden_dim,
                use_leaf_rnn=use_leaf_rnn,
                pooling_method=pooling_method,
                bidirectional=bidirectional,
            )
        elif encoder_type == 'lstm':
            self.encoder = LinearLSTMEncoder(
                word_dim=word_dim, hidden_dim=hidden_dim,
                bidirectional=bidirectional,
                pooling_method=pooling_method
            )
        if bidirectional:
            clf_input_dim = 2 * hidden_dim
        else:
            clf_input_dim = hidden_dim
        self.classifier = Classifier(
            num_classes=num_classes, input_dim=clf_input_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, sents, lengths, fixed_masks=None, return_select_masks=False):
        embeddings = self.dropout(self.word_embedding(sents))
        if self.encoder_type == 'gumbel':
            sents_info = self.encoder(
                inp=embeddings, length=lengths, return_select_masks=return_select_masks
            )
        elif self.encoder_type in ['parsing', 'balanced', 'left', 'right']:
            sents_info = self.encoder(
                inp=embeddings, length=lengths, fixed_masks=fixed_masks
            )
        elif self.encoder_type == 'lstm':
            sents_info = self.encoder(
                inp=embeddings, length=lengths
            )
        else:
            raise Exception('Encoder type {:s} not supported'.format(self.encoder_type))
        logits = self.classifier(sents_info[0])
        if not return_select_masks:
            return logits
        else:
            return logits, sents_info

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PairedSentClassModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, pooling_method,
                 use_batchnorm, dropout_prob, bidirectional, encoder_type,
                 use_leaf_rnn=False):
        super(PairedSentClassModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.pooling_method = pooling_method
        self.use_leaf_rnn = use_leaf_rnn
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.encoder_type = encoder_type

        self.word_embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=word_dim)
        if encoder_type == 'gumbel':
            self.encoder = GumbelTreeLSTMEncoder(
                word_dim=word_dim, hidden_dim=hidden_dim,
                use_leaf_rnn=use_leaf_rnn,
                pooling_method=pooling_method,
                gumbel_temperature=1,
                bidirectional=bidirectional,
            )
        elif encoder_type == 'rvnn':
            self.encoder = RecursiveTreeLSTMEncoder(
                word_dim=word_dim, hidden_dim=hidden_dim,
                use_leaf_rnn=use_leaf_rnn,
                pooling_method=pooling_method,
                bidirectional=bidirectional,
            )
        elif encoder_type == 'lstm':
            self.encoder = LinearLSTMEncoder(
                word_dim=word_dim, hidden_dim=hidden_dim,
                bidirectional=bidirectional,
                pooling_method=pooling_method,
            )
        if bidirectional:
            clf_input_dim = 2 * hidden_dim * 4
        else:
            clf_input_dim = hidden_dim * 4
        self.classifier = Classifier(
            num_classes=num_classes, input_dim=clf_input_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, sents, lengths, fixed_masks=(None, None), return_select_masks=False):
        embeddings_1 = self.dropout(self.word_embedding(sents[0]))
        embeddings_2 = self.dropout(self.word_embedding(sents[1]))
        if self.encoder_type == 'gumbel':
            sents_info_1 = self.encoder(
                inp=embeddings_1, length=lengths[0], return_select_masks=return_select_masks
            )
            sents_info_2 = self.encoder(
                inp=embeddings_2, length=lengths[1], return_select_masks=return_select_masks
            )
        elif self.encoder_type == 'rvnn':
            sents_info_1 = self.encoder(
                inp=embeddings_1, length=lengths[0], fixed_masks=fixed_masks[0]
            )
            sents_info_2 = self.encoder(
                inp=embeddings_2, length=lengths[1], fixed_masks=fixed_masks[1]
            )
        elif self.encoder_type == 'lstm':
            sents_info_1 = self.encoder(
                inp=embeddings_1, length=lengths[0]
            )
            sents_info_2 = self.encoder(
                inp=embeddings_2, length=lengths[1]
            )
        else:
            raise Exception('Encoder type {:s} not supported'.format(self.encoder_type))
        encodings_1 = sents_info_1[0]
        encodings_2 = sents_info_2[0]
        logits = self.classifier(torch.cat(
            [encodings_1, encodings_2, encodings_1 - encodings_2, encodings_1 * encodings_2], dim=1))
        if not return_select_masks:
            return logits
        else:
            return logits, (sents_info_1, sents_info_2)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SentenceDecoder(nn.Module):
    def __init__(self, num_words, word_dim, hidden_dim, dropout=0, n_layers=1, rnn_dropout=0):
        super(SentenceDecoder, self).__init__()
        self.embed = nn.Embedding(num_words, word_dim)
        self.rnn = nn.GRU(
            word_dim, hidden_dim, n_layers, dropout=rnn_dropout, bidirectional=False
        )
        self.decoder = nn.Linear(hidden_dim, num_words)
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.normal_(0, 1)
        self.decoder.weight.data.uniform_(-1, 1)

    def forward(self, code, x, lengths):
        x, lengths, inv, code = sort_sentences_by_lengths(x, lengths, code)
        x = self.dropout(self.embed(x))
        h_0 = code.view(-1, x.size(0), self.hidden_dim)
        x = pack_padded_sequence(x, lengths.data.tolist(), batch_first=True)
        hiddens, h_n = self.rnn(x, h_0)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=-1e8)
        probs = self.decoder(hiddens)
        probs = probs.index_select(0, inv).contiguous()
        return probs

    def decode(self, code, init_word_ids, max_length):
        x = Variable(torch.LongTensor(init_word_ids)).view(-1, 1)
        if torch.cuda.is_available():
            x = x.cuda()
        h = code.view(-1, x.size(0), self.hidden_dim)
        decode_result = list()
        for i in range(max_length):
            x = self.embed(x)
            lengths = [1 for _ in range(code.size(0))]
            x = pack_padded_sequence(x, lengths, batch_first=True)
            hiddens, h_new = self.rnn(x, h)
            hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=-1e8)
            probs = self.decoder(hiddens)
            h = h_new
            x = torch.max(probs, dim=2)[1].view(-1, 1)
            decode_result.append(x)
        decode_result = torch.cat(decode_result, dim=1)
        return decode_result

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder_type, num_src_words, num_tgt_words, word_dim, hidden_dim, pooling_method,
                 dropout_prob, bidirectional, use_leaf_rnn=False):
        super(Seq2SeqModel, self).__init__()
        self.encoder_type = encoder_type
        self.word_embedding = nn.Embedding(
            num_embeddings=num_src_words,
            embedding_dim=word_dim)
        if self.encoder_type in ['balance', 'parsing', 'left', 'right']:
            self.encoder = RecursiveTreeLSTMEncoder(
                word_dim=word_dim,
                hidden_dim=hidden_dim,
                use_leaf_rnn=use_leaf_rnn,
                bidirectional=bidirectional,
                pooling_method=pooling_method
            )
        elif self.encoder_type == 'gumbel':
            self.encoder = GumbelTreeLSTMEncoder(
                word_dim=word_dim,
                hidden_dim=hidden_dim,
                use_leaf_rnn=use_leaf_rnn,
                pooling_method=pooling_method,
                gumbel_temperature=1,
                bidirectional=bidirectional,
            )
        elif self.encoder_type == 'lstm':
            self.encoder = LinearLSTMEncoder(
                word_dim=word_dim,
                hidden_dim=hidden_dim,
                pooling_method=pooling_method,
                bidirectional=bidirectional,
            )
        self.decoder = SentenceDecoder(
            num_words=num_tgt_words,
            word_dim=word_dim,
            hidden_dim=hidden_dim * (2 if bidirectional else 1),
            dropout=dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inp, src_lengths, tgt, tgt_lengths, fixed_masks=None, return_select_masks=False):
        embeddings = self.dropout(self.word_embedding(inp))
        if self.encoder_type == 'gumbel':
            sents_info = self.encoder(
                inp=embeddings, length=src_lengths, return_select_masks=return_select_masks
            )
        elif self.encoder_type in ['balance', 'parsing', 'left', 'right']:
            sents_info = self.encoder(
                inp=embeddings, length=src_lengths, fixed_masks=fixed_masks
            )
        elif self.encoder_type == 'lstm':
            sents_info = self.encoder(
                inp=embeddings, length=src_lengths
            )
        else:
            raise Exception('Encoder type {:s} not supported'.format(self.encoder_type))
        code = sents_info[0]
        decode_result = self.decoder(code, tgt, tgt_lengths)
        return decode_result, sents_info

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
