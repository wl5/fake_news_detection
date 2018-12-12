__all__ = ['BERTClassifier']

from mxnet.gluon import Block
from mxnet.gluon import nn

class BERTClassifier(Block):
    """Model for sentence (pair) classification task with BERT.
    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for
    classification.
    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the bert output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """
    def __init__(self, bert, num_classes=2, dropout=0.0, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes, flatten=False))

    def forward(self, inputs, token_types, valid_length=None): # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.
        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.
        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes)
        """
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)
