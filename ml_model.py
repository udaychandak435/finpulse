# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, Bidirectional,
#     GRU, LSTM, Dense, Dropout, BatchNormalization
# )
# from tensorflow.keras.initializers import GlorotUniform, RandomNormal, Constant
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
# warnings.filterwarnings("ignore")

# # ─── Reproducibility seeds ────────────────────────────────────────────────────
# SEED = 42
# tf.random.set_seed(SEED)
# np.random.seed(SEED)

# # ─── Initializers ─────────────────────────────────────────────────────────────
# k_init = GlorotUniform(seed=SEED)
# r_init = RandomNormal(mean=0.0, stddev=0.1, seed=SEED)
# b_init = Constant(value=0.0)


# def build_model(seq_len: int, n_features: int, n_outputs: int = 4) -> Model:
#     """
#     CNN → Bidirectional GRU → LSTM → Dense (OHLC prediction).
#     Architecture:
#         Input → Conv1D → MaxPool → BiGRU → LSTM → Dense → Output
#     """
#     inputs = Input(shape=(seq_len, n_features), name="input_seq")

#     # ── Block 1: Temporal convolution ──────────────────────────────────────
#     x = Conv1D(
#         filters=64, kernel_size=3, padding="same",
#         activation="relu", kernel_initializer=k_init,
#         bias_initializer=b_init, name="conv1d_1"
#     )(inputs)
#     x = BatchNormalization()(x)
#     x = Conv1D(
#         filters=128, kernel_size=3, padding="same",
#         activation="relu", kernel_initializer=k_init,
#         bias_initializer=b_init, name="conv1d_2"
#     )(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
#     x = Dropout(0.2)(x)

#     # ── Block 2: Bidirectional GRU ─────────────────────────────────────────
#     x = Bidirectional(
#         GRU(
#             units=128, return_sequences=True,
#             kernel_initializer=k_init,
#             recurrent_initializer=r_init,
#             bias_initializer=b_init,
#         ),
#         name="bigru_1"
#     )(x)
#     x = Dropout(0.2)(x)
#     x = Bidirectional(
#         GRU(
#             units=64, return_sequences=True,
#             kernel_initializer=k_init,
#             recurrent_initializer=r_init,
#             bias_initializer=b_init,
#         ),
#         name="bigru_2"
#     )(x)
#     x = Dropout(0.2)(x)

#     # ── Block 3: LSTM ──────────────────────────────────────────────────────
#     x = LSTM(
#         units=64, return_sequences=False,
#         kernel_initializer=k_init,
#         recurrent_initializer=r_init,
#         bias_initializer=b_init,
#         name="lstm_1"
#     )(x)
#     x = Dropout(0.2)(x)

#     # ── Block 4: Dense output ──────────────────────────────────────────────
#     x = Dense(64, activation="relu", kernel_initializer=k_init,
#                bias_initializer=b_init)(x)
#     x = Dense(32, activation="relu", kernel_initializer=k_init,
#                bias_initializer=b_init)(x)
#     outputs = Dense(n_outputs, activation="linear", name="ohlc_output")(x)

#     model = Model(inputs=inputs, outputs=outputs, name="FinPulse_CNN_BiGRU_LSTM")
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#                   loss="huber", metrics=["mae"])
#     return model


# def train_model(X_train, y_train, X_val, y_val, seq_len: int, n_features: int):
#     """
#     Build and train the model. Returns (trained_model, history).
#     """
#     model = build_model(seq_len=seq_len, n_features=n_features)

#     callbacks = [
#         EarlyStopping(monitor="val_loss", patience=15,
#                       restore_best_weights=True, verbose=0),
#         ReduceLROnPlateau(monitor="val_loss", factor=0.5,
#                           patience=7, min_lr=1e-6, verbose=0),
#     ]

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=100,
#         batch_size=32,
#         callbacks=callbacks,
#         verbose=0,
#         shuffle=False,         # temporal order matters
#     )
#     return model, history


# def predict_next_day(model, last_sequence: np.ndarray, scaler_y) -> np.ndarray:
#     """
#     Predict next-day OHLC given the last sequence window.
#     last_sequence shape: (1, seq_len, n_features)
#     Returns unnormalized OHLC array of shape (4,).
#     """
#     pred_scaled = model.predict(last_sequence, verbose=0)
#     pred_ohlc   = scaler_y.inverse_transform(pred_scaled)
#     return pred_ohlc[0]   # [Open, High, Low, Close]







# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, Bidirectional,
#     GRU, LSTM, Dense, Dropout, BatchNormalization
# )
# from tensorflow.keras.initializers import GlorotUniform, RandomNormal, Constant
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
# warnings.filterwarnings("ignore")

# SEED = 42
# tf.random.set_seed(SEED)
# np.random.seed(SEED)

# k_init = GlorotUniform(seed=SEED)
# r_init = RandomNormal(mean=0.0, stddev=0.1, seed=SEED)
# b_init = Constant(value=0.0)


# def build_model(
#     seq_len:      int,
#     n_features:   int,
#     n_outputs:    int   = 4,
#     lstm_units:   int   = 64,
#     gru_units:    int   = 64,
#     conv_filters: int   = 64,
#     dropout:      float = 0.20,
# ) -> Model:
#     """
#     CNN → Bidirectional GRU → LSTM → Dense (OHLC prediction).
#     All architectural hyperparameters are configurable per stock.
#     """
#     inputs = Input(shape=(seq_len, n_features), name="input_seq")

#     # Block 1: Temporal Convolution
#     x = Conv1D(
#         filters=conv_filters, kernel_size=3, padding="same",
#         activation="relu", kernel_initializer=k_init,
#         bias_initializer=b_init, name="conv1d_1"
#     )(inputs)
#     x = BatchNormalization()(x)
#     x = Conv1D(
#         filters=conv_filters * 2, kernel_size=3, padding="same",
#         activation="relu", kernel_initializer=k_init,
#         bias_initializer=b_init, name="conv1d_2"
#     )(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
#     x = Dropout(dropout)(x)

#     # Block 2: Bidirectional GRU
#     x = Bidirectional(
#         GRU(
#             units=gru_units, return_sequences=True,
#             kernel_initializer=k_init,
#             recurrent_initializer=r_init,
#             bias_initializer=b_init,
#         ),
#         name="bigru_1"
#     )(x)
#     x = Dropout(dropout)(x)
#     x = Bidirectional(
#         GRU(
#             units=max(gru_units // 2, 16), return_sequences=True,
#             kernel_initializer=k_init,
#             recurrent_initializer=r_init,
#             bias_initializer=b_init,
#         ),
#         name="bigru_2"
#     )(x)
#     x = Dropout(dropout)(x)

#     # Block 3: LSTM
#     x = LSTM(
#         units=lstm_units, return_sequences=False,
#         kernel_initializer=k_init,
#         recurrent_initializer=r_init,
#         bias_initializer=b_init,
#         name="lstm_1"
#     )(x)
#     x = Dropout(dropout)(x)

#     # Block 4: Dense Output
#     x = Dense(lstm_units,               activation="relu",
#                kernel_initializer=k_init, bias_initializer=b_init)(x)
#     x = Dense(max(lstm_units // 2, 16), activation="relu",
#                kernel_initializer=k_init, bias_initializer=b_init)(x)
#     outputs = Dense(n_outputs, activation="linear", name="ohlc_output")(x)

#     model = Model(inputs=inputs, outputs=outputs, name="FinPulse_CNN_BiGRU_LSTM")
#     return model


# def train_model(
#     X_train,       y_train,
#     X_val,         y_val,
#     seq_len:       int,
#     n_features:    int,
#     lstm_units:    int   = 64,
#     gru_units:     int   = 64,
#     conv_filters:  int   = 64,
#     dropout:       float = 0.20,
#     learning_rate: float = 0.001,
#     epochs:        int   = 100,
#     batch_size:    int   = 32,
# ):
#     """
#     Build and train the model with per-stock hyperparameters.
#     Returns (trained_model, history).
#     """
#     model = build_model(
#         seq_len=seq_len,
#         n_features=n_features,
#         lstm_units=lstm_units,
#         gru_units=gru_units,
#         conv_filters=conv_filters,
#         dropout=dropout,
#     )
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss="huber",
#         metrics=["mae"],
#     )

#     callbacks = [
#         EarlyStopping(
#             monitor="val_loss", patience=15,
#             restore_best_weights=True, verbose=0
#         ),
#         ReduceLROnPlateau(
#             monitor="val_loss", factor=0.5,
#             patience=7, min_lr=1e-6, verbose=0
#         ),
#     ]

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         verbose=0,
#         shuffle=False,
#     )
#     return model, history


# def predict_next_day(model, last_sequence: np.ndarray, scaler_y) -> np.ndarray:
#     """
#     Predict next-day OHLC.
#     last_sequence shape: (1, seq_len, n_features)
#     Returns unnormalized OHLC array of shape (4,).
#     """
#     pred_scaled = model.predict(last_sequence, verbose=0)
#     pred_ohlc   = scaler_y.inverse_transform(pred_scaled)
#     return pred_ohlc[0]





# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, Bidirectional,
#     GRU, LSTM, Dense, Dropout, BatchNormalization,
#     Concatenate, Layer
# )
# from tensorflow.keras.initializers import GlorotUniform, RandomNormal, Constant
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
# warnings.filterwarnings("ignore")

# SEED = 42
# tf.random.set_seed(SEED)
# np.random.seed(SEED)

# k_init = GlorotUniform(seed=SEED)
# r_init = RandomNormal(mean=0.0, stddev=0.1, seed=SEED)
# b_init = Constant(value=0.0)


# class BahdanauAttention(Layer):
#     """
#     Additive attention over BiGRU hidden states.
#     Prevents flat prediction lines by weighting relevant recent timesteps
#     instead of summarising all timesteps equally.
#     """
#     def __init__(self, units, **kwargs):
#         super().__init__(**kwargs)
#         self.units = units
#         self.W     = Dense(units)
#         self.V     = Dense(1)

#     def call(self, encoder_output):
#         score   = self.V(tf.nn.tanh(self.W(encoder_output)))  # (batch, T, 1)
#         weights = tf.nn.softmax(score, axis=1)                 # (batch, T, 1)
#         context = tf.reduce_sum(weights * encoder_output, axis=1)  # (batch, F)
#         return context

#     def get_config(self):
#         config = super().get_config()
#         config.update({"units": self.units})
#         return config


# def combined_loss(y_true, y_pred):
#     """
#     Huber loss + directional penalty.
#     Huber handles outliers; directional term penalises predicting
#     the wrong trend direction — prevents the flat-line problem.
#     """
#     huber     = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
#     true_sign = tf.sign(y_true)
#     pred_sign = tf.sign(y_pred)
#     dir_loss  = tf.reduce_mean(
#         tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
#     )
#     return huber + 0.2 * dir_loss


# def build_model(
#     seq_len:      int,
#     n_features:   int,
#     n_outputs:    int   = 4,
#     lstm_units:   int   = 64,
#     gru_units:    int   = 64,
#     conv_filters: int   = 64,
#     dropout:      float = 0.20,
# ) -> Model:
#     """
#     Architecture:
#     Input → Conv1D × 2 → MaxPool → BiGRU × 2 → [Attention || LSTM] → Merge → Dense → OHLC
#     """
#     inputs = Input(shape=(seq_len, n_features), name="input_seq")

#     # ── Block 1: Temporal Convolution ─────────────────────────────────────────
#     x = Conv1D(filters=conv_filters, kernel_size=3, padding="same",
#                activation="relu", kernel_initializer=k_init,
#                bias_initializer=b_init, name="conv1d_1")(inputs)
#     x = BatchNormalization()(x)
#     x = Conv1D(filters=conv_filters * 2, kernel_size=3, padding="same",
#                activation="relu", kernel_initializer=k_init,
#                bias_initializer=b_init, name="conv1d_2")(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
#     x = Dropout(dropout)(x)

#     # ── Block 2: Bidirectional GRU ────────────────────────────────────────────
#     x = Bidirectional(
#         GRU(gru_units, return_sequences=True,
#             kernel_initializer=k_init,
#             recurrent_initializer=r_init,
#             bias_initializer=b_init),
#         name="bigru_1"
#     )(x)
#     x = Dropout(dropout)(x)
#     x = Bidirectional(
#         GRU(max(gru_units // 2, 16), return_sequences=True,
#             kernel_initializer=k_init,
#             recurrent_initializer=r_init,
#             bias_initializer=b_init),
#         name="bigru_2"
#     )(x)
#     x = Dropout(dropout)(x)

#     # ── Block 3a: Bahdanau Attention ──────────────────────────────────────────
#     attention_context = BahdanauAttention(units=gru_units, name="attention")(x)

#     # ── Block 3b: LSTM (parallel path) ───────────────────────────────────────
#     x_lstm = LSTM(
#         units=lstm_units, return_sequences=False,
#         kernel_initializer=k_init,
#         recurrent_initializer=r_init,
#         bias_initializer=b_init,
#         name="lstm_1"
#     )(x)

#     # ── Block 4: Merge attention context + LSTM output ────────────────────────
#     merged = Concatenate(name="attn_lstm_merge")([x_lstm, attention_context])
#     merged = Dropout(dropout)(merged)

#     # ── Block 5: Dense output ─────────────────────────────────────────────────
#     x_out  = Dense(lstm_units, activation="relu",
#                    kernel_initializer=k_init, bias_initializer=b_init)(merged)
#     x_out  = Dense(max(lstm_units // 2, 16), activation="relu",
#                    kernel_initializer=k_init, bias_initializer=b_init)(x_out)
#     outputs = Dense(n_outputs, activation="linear", name="ohlc_output")(x_out)

#     return Model(inputs=inputs, outputs=outputs, name="FinPulse_Attn_CNN_BiGRU_LSTM")


# def train_model(
#     X_train, y_train,
#     X_val,   y_val,
#     seq_len:       int,
#     n_features:    int,
#     lstm_units:    int   = 64,
#     gru_units:     int   = 64,
#     conv_filters:  int   = 64,
#     dropout:       float = 0.20,
#     learning_rate: float = 0.001,
#     epochs:        int   = 100,
#     batch_size:    int   = 32,
# ):
#     """
#     Build and train model.
#     run_eagerly=True — disables tf.function graph compilation per thread,
#     which prevents retracing warnings and Streamlit silent hangs.
#     """
#     model = build_model(
#         seq_len=seq_len, n_features=n_features,
#         lstm_units=lstm_units, gru_units=gru_units,
#         conv_filters=conv_filters, dropout=dropout,
#     )
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=learning_rate,
#             clipnorm=1.0,
#         ),
#         loss=combined_loss,
#         metrics=["mae"],
#         run_eagerly=True,   # prevents tf.function retracing across threads
#     )

#     callbacks = [
#         EarlyStopping(monitor="val_loss", patience=15,
#                       restore_best_weights=True, verbose=0),
#         ReduceLROnPlateau(monitor="val_loss", factor=0.5,
#                           patience=7, min_lr=1e-6, verbose=0),
#     ]

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         verbose=0,
#         shuffle=False,
#     )
#     return model, history


# def predict_next_day(model, last_sequence: np.ndarray, scaler_y) -> np.ndarray:
#     """
#     Predict next-day OHLC.
#     last_sequence shape: (1, seq_len, n_features)
#     Returns unnormalized OHLC array of shape (4,).
#     """
#     pred_scaled = model.predict(last_sequence, verbose=0)
#     pred_ohlc   = scaler_y.inverse_transform(pred_scaled)
#     return pred_ohlc[0]







# # ml_model-5.py  ──  PyTorch GPU rewrite (drop-in replacement for TensorFlow version)
# # Architecture: Conv1D x2 → MaxPool → BiGRU x2 → Bahdanau Attention ║ LSTM → Merge → Dense → OHLC

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import warnings
# warnings.filterwarnings("ignore")

# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[FinPulse] Using device: {DEVICE} "
#       f"({'RTX 4060 🚀' if torch.cuda.is_available() else 'CPU ⚠️'})")

# # ── Autocast dtype for mixed precision (AMP) ──────────────────────────────────
# AMP_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# # ─────────────────────────────────────────────────────────────────────────────
# # Bahdanau (Additive) Attention
# # ─────────────────────────────────────────────────────────────────────────────
# class BahdanauAttention(nn.Module):
#     """Additive attention over BiGRU hidden states.
#     Weights recent timesteps that matter most — prevents flat prediction lines."""
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.W = nn.Linear(hidden_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, 1)

#     def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
#         # encoder_output: (batch, T, hidden_dim)
#         score = self.V(torch.tanh(self.W(encoder_output)))   # (batch, T, 1)
#         weights = torch.softmax(score, dim=1)                # (batch, T, 1)
#         context = (weights * encoder_output).sum(dim=1)      # (batch, hidden_dim)
#         return context


# # ─────────────────────────────────────────────────────────────────────────────
# # Full Model: Conv1D → BiGRU → Attention ║ LSTM → Dense → OHLC
# # ─────────────────────────────────────────────────────────────────────────────
# class FinPulseModel(nn.Module):
#     def __init__(
#         self,
#         n_features: int,
#         seq_len: int,
#         n_outputs: int = 4,
#         lstm_units: int = 64,
#         gru_units: int = 64,
#         conv_filters: int = 64,
#         dropout: float = 0.20,
#     ):
#         super().__init__()

#         # ── Block 1: Temporal Convolution ─────────────────────────────────────
#         # Conv1D in PyTorch: input shape (batch, channels, seq_len)
#         self.conv1 = nn.Conv1d(n_features, conv_filters, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm1d(conv_filters)
#         self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm1d(conv_filters * 2)
#         self.pool  = nn.MaxPool1d(kernel_size=2)
#         self.drop1 = nn.Dropout(dropout)

#         conv_out_len = seq_len // 2          # after MaxPool(2)
#         conv_out_ch  = conv_filters * 2

#         # ── Block 2: Bidirectional GRU ─────────────────────────────────────────
#         self.bigru1 = nn.GRU(conv_out_ch, gru_units,
#                              batch_first=True, bidirectional=True)
#         self.drop2  = nn.Dropout(dropout)
#         gru2_in     = gru_units * 2          # bidirectional doubles output dim
#         gru2_units  = max(gru_units * 2, 16) // 2   # mirrors Keras: max(gru*2,16) but //2 for each direction
#         self.bigru2 = nn.GRU(gru2_in, gru2_units,
#                              batch_first=True, bidirectional=True)
#         self.drop3  = nn.Dropout(dropout)
#         bigru2_out  = gru2_units * 2

#         # ── Block 3a: Bahdanau Attention (on BiGRU output) ─────────────────────
#         self.attention = BahdanauAttention(bigru2_out)

#         # ── Block 3b: LSTM parallel path ───────────────────────────────────────
#         # Receives the full BiGRU sequence, outputs last hidden state
#         self.lstm = nn.LSTM(bigru2_out, lstm_units, batch_first=True)
#         self.drop4 = nn.Dropout(dropout)

#         # ── Block 4: Merge Attention + LSTM ────────────────────────────────────
#         merge_dim = lstm_units + bigru2_out
#         self.fc1  = nn.Linear(merge_dim, lstm_units)
#         self.fc2  = nn.Linear(lstm_units, max(lstm_units // 2, 16))
#         self.out  = nn.Linear(max(lstm_units // 2, 16), n_outputs)
#         self.relu = nn.ReLU()

#         # Xavier init (mirrors Glorot in Keras)
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, (nn.GRU, nn.LSTM)):
#                 for name, p in m.named_parameters():
#                     if "weight_ih" in name:
#                         nn.init.xavier_uniform_(p)
#                     elif "weight_hh" in name:
#                         nn.init.orthogonal_(p)
#                     elif "bias" in name:
#                         nn.init.zeros_(p)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch, seq_len, n_features)

#         # Conv expects (batch, channels, seq_len) → transpose
#         xc = x.permute(0, 2, 1)
#         xc = self.relu(self.bn1(self.conv1(xc)))
#         xc = self.relu(self.bn2(self.conv2(xc)))
#         xc = self.pool(xc)
#         xc = self.drop1(xc)
#         xc = xc.permute(0, 2, 1)          # back to (batch, seq_len//2, channels)

#         # BiGRU block
#         xg, _ = self.bigru1(xc)
#         xg    = self.drop2(xg)
#         xg, _ = self.bigru2(xg)
#         xg    = self.drop3(xg)             # (batch, T', bigru2_out)

#         # Attention context
#         ctx  = self.attention(xg)          # (batch, bigru2_out)

#         # LSTM parallel path
#         xl, (hn, _) = self.lstm(xg)
#         xl = hn[-1]                        # last hidden state (batch, lstm_units)
#         xl = self.drop4(xl)

#         # Merge + Dense
#         merged = torch.cat([xl, ctx], dim=1)
#         out    = self.relu(self.fc1(merged))
#         out    = self.relu(self.fc2(out))
#         out    = self.out(out)             # (batch, 4) → OHLC
#         return out


# # ─────────────────────────────────────────────────────────────────────────────
# # Combined Loss: Huber + Directional Penalty
# # ─────────────────────────────────────────────────────────────────────────────
# def combined_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#     """Huber loss + directional penalty.
#     Penalises predicting the wrong trend direction — prevents flat-line outputs."""
#     huber    = nn.HuberLoss(delta=1.0)(y_pred, y_true)
#     dir_loss = ((y_true.sign() != y_pred.sign()).float().mean())
#     return huber + 0.2 * dir_loss


# # ─────────────────────────────────────────────────────────────────────────────
# # train_model  ── drop-in replacement, same signature as TF version
# # ─────────────────────────────────────────────────────────────────────────────
# def train_model(
#     X_train, y_train, X_val, y_val,
#     seq_len: int,
#     n_features: int,
#     lstm_units: int = 64,
#     gru_units: int  = 64,
#     conv_filters: int = 64,
#     dropout: float  = 0.20,
#     learning_rate: float = 0.001,
#     epochs: int     = 100,
#     batch_size: int = 32,
# ):
#     """Build, train, and return a FinPulseModel on GPU (or CPU fallback).
#     Returns (model, history_dict) — history mirrors Keras History object keys."""

#     # Convert numpy → tensors, send to GPU
#     Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
#     yt = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
#     Xv = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
#     yv = torch.tensor(y_val,   dtype=torch.float32).to(DEVICE)

#     train_loader = DataLoader(
#         TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False
#     )

#     model = FinPulseModel(
#         n_features=n_features, seq_len=seq_len,
#         lstm_units=lstm_units, gru_units=gru_units,
#         conv_filters=conv_filters, dropout=dropout,
#     ).to(DEVICE)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
#     optimizer.param_groups[0]["max_norm"] = 1.0  # mirrors clipnorm=1.0

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
#     )

#     # Mixed-precision scaler (only active when CUDA available)
#     scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

#     # EarlyStopping state
#     best_val_loss   = float("inf")
#     best_state_dict = None
#     patience_count  = 0
#     PATIENCE        = 15

#     history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

#     for epoch in range(epochs):
#         # ── Training ──────────────────────────────────────────────────────────
#         model.train()
#         train_loss, train_mae = 0.0, 0.0
#         for xb, yb in train_loader:
#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(),
#                                          dtype=AMP_DTYPE):
#                 pred = model(xb)
#                 loss = combined_loss(pred, yb)
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             train_loss += loss.item()
#             train_mae  += (pred - yb).abs().mean().item()

#         train_loss /= len(train_loader)
#         train_mae  /= len(train_loader)

#         # ── Validation ────────────────────────────────────────────────────────
#         model.eval()
#         with torch.no_grad(), torch.cuda.amp.autocast(
#             enabled=torch.cuda.is_available(), dtype=AMP_DTYPE
#         ):
#             val_pred = model(Xv)
#             val_loss = combined_loss(val_pred, yv).item()
#             val_mae  = (val_pred - yv).abs().mean().item()

#         scheduler.step(val_loss)
#         history["loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["mae"].append(train_mae)
#         history["val_mae"].append(val_mae)

#         # EarlyStopping
#         if val_loss < best_val_loss - 1e-6:
#             best_val_loss   = val_loss
#             best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             patience_count  = 0
#         else:
#             patience_count += 1
#             if patience_count >= PATIENCE:
#                 break  # restore_best_weights equivalent below

#     # Restore best weights (mirrors EarlyStopping restore_best_weights=True)
#     if best_state_dict is not None:
#         model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_dict.items()})

#     return model, history


# # ─────────────────────────────────────────────────────────────────────────────
# # predict_next_day  ── drop-in replacement, same signature as TF version
# # ─────────────────────────────────────────────────────────────────────────────
# def predict_next_day(model, last_sequence: np.ndarray, scaler_y) -> np.ndarray:
#     """Predict next-day OHLC.
#     last_sequence shape: (1, seq_len, n_features)
#     Returns unnormalized OHLC array of shape (4,)."""
#     model.eval()
#     with torch.no_grad():
#         x = torch.tensor(last_sequence, dtype=torch.float32).to(DEVICE)
#         pred_scaled = model(x).cpu().numpy()   # (1, 4)
#     pred_ohlc = scaler_y.inverse_transform(pred_scaled)
#     return pred_ohlc[0]









# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import math
# import warnings
# warnings.filterwarnings("ignore")

# # ─────────────────────────────────────────────────────────────────────────────
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[ml_model_torch] Using device: {DEVICE}")


# # ══════════════════════════════════════════════════════════════════════════════
# # Positional Encoding
# # ══════════════════════════════════════════════════════════════════════════════
# class PositionalEncoding(nn.Module):
#     """
#     Sinusoidal positional encoding.
#     PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#     Shape: (1, max_len, d_model) — added to input embeddings.
#     """
#     def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe       = torch.zeros(max_len, d_model)                    # (max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()    # (max_len, 1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)                                         # (1, max_len, d_model)
#         self.register_buffer("pe", pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch, seq_len, d_model)
#         x = x + self.pe[:, : x.size(1), :]
#         return self.dropout(x)


# # ══════════════════════════════════════════════════════════════════════════════
# # Transformer Encoder Block
# # ══════════════════════════════════════════════════════════════════════════════
# class TransformerEncoderBlock(nn.Module):
#     """
#     Standard Transformer encoder:
#       x → MultiHeadAttention → Add & Norm → FFN → Add & Norm
#     Captures long-range temporal dependencies.
#     """
#     def __init__(self, d_model: int, num_heads: int,
#                  ff_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.self_attn  = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=num_heads,
#             dropout=dropout, batch_first=True
#         )
#         self.ffn        = nn.Sequential(
#             nn.Linear(d_model, ff_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_dim, d_model),
#         )
#         self.norm1      = nn.LayerNorm(d_model, eps=1e-6)
#         self.norm2      = nn.LayerNorm(d_model, eps=1e-6)
#         self.dropout1   = nn.Dropout(dropout)
#         self.dropout2   = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Self-attention + residual
#         attn_out, _ = self.self_attn(x, x, x)
#         x           = self.norm1(x + self.dropout1(attn_out))
#         # FFN + residual
#         ffn_out     = self.ffn(x)
#         x           = self.norm2(x + self.dropout2(ffn_out))
#         return x


# # ══════════════════════════════════════════════════════════════════════════════
# # Bahdanau Attention (over BiGRU hidden states)
# # ══════════════════════════════════════════════════════════════════════════════
# class BahdanauAttention(nn.Module):
#     """
#     Additive attention over a sequence of hidden states.
#     context = Σ softmax(V · tanh(W · h_t)) · h_t
#     Focuses on relevant recent BiGRU timesteps.
#     """
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.V = nn.Linear(hidden_dim, 1,           bias=False)

#     def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
#         # encoder_output: (batch, seq_len, hidden_dim)
#         score   = self.V(torch.tanh(self.W(encoder_output)))   # (batch, seq_len, 1)
#         weights = torch.softmax(score, dim=1)                   # (batch, seq_len, 1)
#         context = (weights * encoder_output).sum(dim=1)         # (batch, hidden_dim)
#         return context


# # ══════════════════════════════════════════════════════════════════════════════
# # Hybrid Model
# # ══════════════════════════════════════════════════════════════════════════════
# class FinPulseHybrid(nn.Module):
#     """
#     Hybrid Architecture:
#     ┌─────────────────────────────────────────────────────┐
#     │  Input (batch, seq_len, n_features)                 │
#     │       ↓                                             │
#     │  ┌────────────────┐   ┌──────────────────────────┐  │
#     │  │  CNN Branch    │   │  Transformer Branch      │  │
#     │  │  Conv1d × 2    │   │  Linear Proj → PosEnc    │  │
#     │  │  MaxPool       │   │  TransformerBlock × N    │  │
#     │  │  BiGRU × 2     │   │  GlobalAvgPool            │  │
#     │  │  BahdanauAttn  │   └──────────────────────────┘  │
#     │  │  LSTM          │              ↓                   │
#     │  └────────────────┘         Concat                   │
#     │         ↓                      ↓                    │
#     │         └──────── Concat ──────┘                    │
#     │                     ↓                               │
#     │              Dense → Dense → Log-Returns (4)        │
#     └─────────────────────────────────────────────────────┘
#     """
#     def __init__(
#         self,
#         seq_len:              int,
#         n_features:           int,
#         n_outputs:            int   = 4,
#         lstm_units:           int   = 64,
#         gru_units:            int   = 64,
#         conv_filters:         int   = 64,
#         dropout:              float = 0.20,
#         d_model:              int   = 64,
#         num_heads:            int   = 4,
#         ff_dim:               int   = 128,
#         n_transformer_blocks: int   = 2,
#     ):
#         super().__init__()
#         self.seq_len    = seq_len
#         self.n_features = n_features

#         # ── Branch A: CNN ─────────────────────────────────────────────────────
#         # Conv1d expects (batch, channels, seq_len) — we'll permute in forward
#         self.conv1      = nn.Conv1d(n_features,       conv_filters,     3, padding=1)
#         self.bn1        = nn.BatchNorm1d(conv_filters)
#         self.conv2      = nn.Conv1d(conv_filters,     conv_filters * 2, 3, padding=1)
#         self.bn2        = nn.BatchNorm1d(conv_filters * 2)
#         self.pool       = nn.MaxPool1d(kernel_size=2)
#         self.drop_conv  = nn.Dropout(dropout)

#         conv_out_dim    = conv_filters * 2
#         conv_seq_len    = seq_len // 2   # after MaxPool(2)

#         # ── Branch A: BiGRU ───────────────────────────────────────────────────
#         self.bigru1     = nn.GRU(
#             input_size=conv_out_dim, hidden_size=gru_units,
#             num_layers=1, batch_first=True, bidirectional=True
#         )
#         self.drop_gru1  = nn.Dropout(dropout)
#         gru2_in         = gru_units * 2   # bidirectional doubles output

#         gru2_units      = max(gru_units // 2, 16)
#         self.bigru2     = nn.GRU(
#             input_size=gru2_in, hidden_size=gru2_units,
#             num_layers=1, batch_first=True, bidirectional=True
#         )
#         self.drop_gru2  = nn.Dropout(dropout)
#         gru2_out_dim    = gru2_units * 2

#         # ── Branch A: Bahdanau Attention ──────────────────────────────────────
#         self.bahdanau   = BahdanauAttention(hidden_dim=gru2_out_dim)

#         # ── Branch A: LSTM ────────────────────────────────────────────────────
#         self.lstm       = nn.LSTM(
#             input_size=gru2_out_dim, hidden_size=lstm_units,
#             num_layers=1, batch_first=True
#         )
#         self.drop_lstm  = nn.Dropout(dropout)

#         branch_a_dim    = lstm_units + gru2_out_dim   # LSTM + attention context

#         # ── Branch B: Transformer ─────────────────────────────────────────────
#         self.input_proj = nn.Linear(n_features, d_model)
#         self.pos_enc    = PositionalEncoding(d_model=d_model,
#                                              max_len=seq_len + 10,
#                                              dropout=dropout)
#         self.transformer_blocks = nn.ModuleList([
#             TransformerEncoderBlock(
#                 d_model=d_model, num_heads=num_heads,
#                 ff_dim=ff_dim, dropout=dropout
#             )
#             for _ in range(n_transformer_blocks)
#         ])
#         self.drop_transformer = nn.Dropout(dropout)
#         branch_b_dim    = d_model

#         # ── Merge + Dense Head ────────────────────────────────────────────────
#         merged_dim      = branch_a_dim + branch_b_dim
#         head_dim        = lstm_units + d_model

#         self.dense1     = nn.Linear(merged_dim, head_dim)
#         self.drop_d1    = nn.Dropout(dropout)
#         self.dense2     = nn.Linear(head_dim, max(head_dim // 2, 32))
#         self.out        = nn.Linear(max(head_dim // 2, 32), n_outputs)

#         self._init_weights()

#     def _init_weights(self):
#         """Xavier uniform init for all Linear and Conv layers."""
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv1d)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.GRU):
#                 for name, param in m.named_parameters():
#                     if "weight" in name:
#                         nn.init.xavier_uniform_(param)
#                     elif "bias" in name:
#                         nn.init.zeros_(param)
#             elif isinstance(m, nn.LSTM):
#                 for name, param in m.named_parameters():
#                     if "weight" in name:
#                         nn.init.xavier_uniform_(param)
#                     elif "bias" in name:
#                         nn.init.zeros_(param)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (batch, seq_len, n_features)
#         returns: (batch, 4) — predicted log returns for OHLC
#         """
#         # ── Branch A: CNN ─────────────────────────────────────────────────────
#         # Conv1d needs (batch, channels, seq_len)
#         a = x.permute(0, 2, 1)                          # (B, F, T)
#         a = F.relu(self.bn1(self.conv1(a)))              # (B, conv_filters, T)
#         a = F.relu(self.bn2(self.conv2(a)))              # (B, conv_filters*2, T)
#         a = self.pool(a)                                  # (B, conv_filters*2, T//2)
#         a = self.drop_conv(a)
#         a = a.permute(0, 2, 1)                          # (B, T//2, conv_filters*2)

#         # ── Branch A: BiGRU ───────────────────────────────────────────────────
#         a, _ = self.bigru1(a)                            # (B, T//2, gru_units*2)
#         a    = self.drop_gru1(a)
#         a, _ = self.bigru2(a)                            # (B, T//2, gru2_units*2)
#         a    = self.drop_gru2(a)

#         # ── Branch A: Bahdanau + LSTM ─────────────────────────────────────────
#         attn_ctx     = self.bahdanau(a)                  # (B, gru2_out_dim)
#         lstm_out, _  = self.lstm(a)                      # (B, T//2, lstm_units)
#         lstm_last    = lstm_out[:, -1, :]                # (B, lstm_units)
#         lstm_last    = self.drop_lstm(lstm_last)

#         branch_a = torch.cat([lstm_last, attn_ctx], dim=1)  # (B, branch_a_dim)

#         # ── Branch B: Transformer ─────────────────────────────────────────────
#         b = self.input_proj(x)                           # (B, T, d_model)
#         b = self.pos_enc(b)                              # (B, T, d_model)
#         for block in self.transformer_blocks:
#             b = block(b)                                 # (B, T, d_model)
#         branch_b = b.mean(dim=1)                         # (B, d_model) — global avg pool
#         branch_b = self.drop_transformer(branch_b)

#         # ── Merge ─────────────────────────────────────────────────────────────
#         merged = torch.cat([branch_a, branch_b], dim=1)  # (B, merged_dim)

#         # ── Dense Head ────────────────────────────────────────────────────────
#         out = F.relu(self.dense1(merged))
#         out = self.drop_d1(out)
#         out = F.relu(self.dense2(out))
#         out = self.out(out)                              # (B, 4)
#         return out


# # ══════════════════════════════════════════════════════════════════════════════
# # Combined Loss
# # ══════════════════════════════════════════════════════════════════════════════
# class CombinedLoss(nn.Module):
#     """
#     Huber loss on log-return targets + directional penalty.
#     huber_delta=0.5 — less sensitive to outliers than MSE.
#     dir_weight=0.3  — penalises wrong direction (up/down) prediction.
#     """
#     def __init__(self, huber_delta: float = 0.5, dir_weight: float = 0.3):
#         super().__init__()
#         self.huber     = nn.HuberLoss(delta=huber_delta, reduction="mean")
#         self.dir_weight = dir_weight

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         huber    = self.huber(y_pred, y_true)
#         dir_loss = (torch.sign(y_pred) != torch.sign(y_true)).float().mean()
#         return huber + self.dir_weight * dir_loss


# # ══════════════════════════════════════════════════════════════════════════════
# # Early Stopping (manual — PyTorch has no built-in)
# # ══════════════════════════════════════════════════════════════════════════════
# class EarlyStopping:
#     def __init__(self, patience: int = 15, min_delta: float = 1e-6):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.best_loss  = float("inf")
#         self.counter    = 0
#         self.best_state = None

#     def step(self, val_loss: float, model: nn.Module) -> bool:
#         """Returns True if training should stop."""
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss  = val_loss
#             self.counter    = 0
#             self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         else:
#             self.counter += 1
#         return self.counter >= self.patience

#     def restore_best(self, model: nn.Module):
#         if self.best_state is not None:
#             model.load_state_dict(self.best_state)


# # ══════════════════════════════════════════════════════════════════════════════
# # Trainer
# # ══════════════════════════════════════════════════════════════════════════════
# def train_model(
#     X_train:    np.ndarray,
#     y_train:    np.ndarray,
#     X_val:      np.ndarray,
#     y_val:      np.ndarray,
#     seq_len:    int,
#     n_features: int,
#     lstm_units:           int   = 64,
#     gru_units:            int   = 64,
#     conv_filters:         int   = 64,
#     dropout:              float = 0.20,
#     learning_rate:        float = 0.001,
#     epochs:               int   = 100,
#     batch_size:           int   = 32,
#     d_model:              int   = 64,
#     num_heads:            int   = 4,
#     ff_dim:               int   = 128,
#     n_transformer_blocks: int   = 2,
# ):
#     """
#     Train the hybrid model on numpy arrays.
#     Returns (trained_model, history_dict).
#     """
#     # Convert to tensors
#     X_tr = torch.tensor(X_train, dtype=torch.float32)
#     y_tr = torch.tensor(y_train, dtype=torch.float32)
#     X_vl = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
#     y_vl = torch.tensor(y_val,   dtype=torch.float32).to(DEVICE)

#     train_loader = DataLoader(
#         TensorDataset(X_tr, y_tr),
#         batch_size=batch_size,
#         shuffle=False,   # no shuffle — temporal order must be preserved
#     )

#     model = FinPulseHybrid(
#         seq_len=seq_len, n_features=n_features,
#         lstm_units=lstm_units, gru_units=gru_units,
#         conv_filters=conv_filters, dropout=dropout,
#         d_model=d_model, num_heads=num_heads,
#         ff_dim=ff_dim, n_transformer_blocks=n_transformer_blocks,
#     ).to(DEVICE)

#     criterion    = CombinedLoss(huber_delta=0.5, dir_weight=0.3)
#     optimizer    = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
#     )
#     early_stop   = EarlyStopping(patience=15)

#     history = {"train_loss": [], "val_loss": [], "val_mae": []}

#     for epoch in range(epochs):
#         # ── Train ─────────────────────────────────────────────────────────────
#         model.train()
#         epoch_loss = 0.0
#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(DEVICE)
#             y_batch = y_batch.to(DEVICE)

#             optimizer.zero_grad()
#             pred   = model(X_batch)
#             loss   = criterion(pred, y_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             epoch_loss += loss.item() * len(X_batch)

#         epoch_loss /= len(X_tr)

#         # ── Validate ──────────────────────────────────────────────────────────
#         model.eval()
#         with torch.no_grad():
#             val_pred = model(X_vl)
#             val_loss = criterion(val_pred, y_vl).item()
#             val_mae  = F.l1_loss(val_pred, y_vl).item()

#         scheduler.step(val_loss)
#         history["train_loss"].append(epoch_loss)
#         history["val_loss"].append(val_loss)
#         history["val_mae"].append(val_mae)

#         if (epoch + 1) % 10 == 0:
#             lr_now = optimizer.param_groups[0]["lr"]
#             print(
#                 f"  Epoch {epoch+1:3d}/{epochs} | "
#                 f"train_loss={epoch_loss:.6f} | "
#                 f"val_loss={val_loss:.6f} | "
#                 f"val_mae={val_mae:.6f} | "
#                 f"lr={lr_now:.2e}"
#             )

#         if early_stop.step(val_loss, model):
#             print(f"  EarlyStopping at epoch {epoch+1} "
#                   f"(best val_loss={early_stop.best_loss:.6f})")
#             break

#     early_stop.restore_best(model)
#     model.eval()
#     return model, history


# # ══════════════════════════════════════════════════════════════════════════════
# # Prediction — Log Return → Reconstruct Price
# # ══════════════════════════════════════════════════════════════════════════════
# def predict_next_day(
#     model:        nn.Module,
#     last_sequence: np.ndarray,
#     scaler_y,
#     current_ohlc: np.ndarray,
# ) -> np.ndarray:
#     """
#     Predict next-day OHLC via log-return reconstruction.

#     last_sequence : (1, seq_len, n_features)  numpy
#     scaler_y      : StandardScaler fitted on log returns
#     current_ohlc  : today's raw OHLC array (4,)
#     Returns       : predicted OHLC prices (4,)
#     """
#     model.eval()
#     with torch.no_grad():
#         x           = torch.tensor(last_sequence, dtype=torch.float32).to(DEVICE)
#         pred_scaled = model(x).cpu().numpy()

#     pred_returns = scaler_y.inverse_transform(pred_scaled)[0]
#     pred_ohlc    = current_ohlc * np.exp(pred_returns)
#     return pred_ohlc


# def reconstruct_prices_from_returns(
#     model:          nn.Module,
#     X_val:          np.ndarray,
#     scaler_y,
#     raw_ohlc_prev:  np.ndarray,
#     batch_size:     int = 64,
# ) -> np.ndarray:
#     """
#     Reconstruct validation set prices from predicted log returns.
#     Used for Predicted vs Actual chart.

#     X_val         : (N, seq_len, n_features)
#     raw_ohlc_prev : (N, 4) — previous day actual OHLC
#     Returns       : (N, 4) — reconstructed predicted prices
#     """
#     model.eval()
#     preds = []

#     with torch.no_grad():
#         for i in range(0, len(X_val), batch_size):
#             x_batch = torch.tensor(
#                 X_val[i : i + batch_size], dtype=torch.float32
#             ).to(DEVICE)
#             preds.append(model(x_batch).cpu().numpy())

#     pred_scaled  = np.vstack(preds)
#     pred_returns = scaler_y.inverse_transform(pred_scaled)
#     pred_prices  = raw_ohlc_prev * np.exp(pred_returns)
#     return pred_prices


# # ══════════════════════════════════════════════════════════════════════════════
# # Model Summary Utility
# # ══════════════════════════════════════════════════════════════════════════════
# def print_model_summary(model: nn.Module, seq_len: int, n_features: int):
#     """Print parameter count per module — equivalent to Keras model.summary()."""
#     total  = sum(p.numel() for p in model.parameters())
#     train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n{'='*60}")
#     print(f"  {model.__class__.__name__}")
#     print(f"  Input shape : (batch, {seq_len}, {n_features})")
#     print(f"{'='*60}")
#     for name, module in model.named_children():
#         params = sum(p.numel() for p in module.parameters())
#         print(f"  {name:<30s}  {params:>10,} params")
#     print(f"{'─'*60}")
#     print(f"  Total params      : {total:>10,}")
#     print(f"  Trainable params  : {train:>10,}")
#     print(f"{'='*60}\n")





# =============================================================================
# ml_model.py — PyTorch  CNN + BiGRU + Bahdanau Attention + LSTM
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"[ml_model] Using device: {DEVICE}")


# =============================================================================
# ARCHITECTURE
# =============================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        score   = self.V(torch.tanh(self.W(encoder_output)))
        weights = torch.softmax(score, dim=1)
        return (weights * encoder_output).sum(dim=1)


class FinPulseModel(nn.Module):
    def __init__(
        self,
        n_features   : int,
        seq_len      : int,
        n_outputs    : int   = 4,
        lstm_units   : int   = 64,
        gru_units    : int   = 64,
        conv_filters : int   = 64,
        dropout      : float = 0.20,
    ):
        super().__init__()

        # Conv block
        self.conv1 = nn.Conv1d(n_features, conv_filters,     kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters*2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(conv_filters * 2)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)
        conv_out   = conv_filters * 2

        # BiGRU block
        self.bigru1 = nn.GRU(conv_out,    gru_units,  batch_first=True, bidirectional=True)
        self.drop2  = nn.Dropout(dropout)
        gru2_units  = max(gru_units * 2, 16) // 2
        self.bigru2 = nn.GRU(gru_units*2, gru2_units, batch_first=True, bidirectional=True)
        self.drop3  = nn.Dropout(dropout)
        bigru2_out  = gru2_units * 2

        # Attention + LSTM parallel
        self.attention = BahdanauAttention(bigru2_out)
        self.lstm      = nn.LSTM(bigru2_out, lstm_units, batch_first=True)
        self.drop4     = nn.Dropout(dropout)

        # Dense head
        merge = lstm_units + bigru2_out
        self.fc1 = nn.Linear(merge, lstm_units)
        self.fc2 = nn.Linear(lstm_units, max(lstm_units // 2, 16))
        self.out = nn.Linear(max(lstm_units // 2, 16), n_outputs)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GRU, nn.LSTM)):
                for name, p in m.named_parameters():
                    if   "weight_ih" in name: nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name: nn.init.orthogonal_(p)
                    elif "bias"      in name: nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.permute(0, 2, 1)
        xc = self.relu(self.bn1(self.conv1(xc)))
        xc = self.relu(self.bn2(self.conv2(xc)))
        xc = self.pool(xc)
        xc = self.drop1(xc).permute(0, 2, 1)

        xg, _ = self.bigru1(xc); xg = self.drop2(xg)
        xg, _ = self.bigru2(xg); xg = self.drop3(xg)

        ctx        = self.attention(xg)
        xl, (hn,_) = self.lstm(xg)
        xl         = self.drop4(hn[-1])

        out = torch.cat([xl, ctx], dim=1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.out(out)


# =============================================================================
# LOSS
# =============================================================================
def _combined_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    huber    = nn.HuberLoss(delta=1.0)(pred, true)
    dir_loss = (true.sign() != pred.sign()).float().mean()
    return huber + 0.2 * dir_loss


# =============================================================================
# TRAIN
# =============================================================================
def train_model(
    X_train, y_train, X_val, y_val,
    seq_len      : int,
    n_features   : int,
    lstm_units   : int   = 64,
    gru_units    : int   = 64,
    conv_filters : int   = 64,
    dropout      : float = 0.20,
    learning_rate: float = 0.001,
    epochs       : int   = 100,
    batch_size   : int   = 32,
    **kwargs,
):
    Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val,   dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)

    model = FinPulseModel(
        n_features=n_features, seq_len=seq_len,
        lstm_units=lstm_units, gru_units=gru_units,
        conv_filters=conv_filters, dropout=dropout,
    ).to(DEVICE)

    optimizer  = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )
    scaler     = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    history    = {"loss": [], "val_loss": []}
    best_loss  = float("inf")
    best_state = None
    patience   = 0
    PATIENCE   = 15

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=AMP_DTYPE):
                loss = _combined_loss(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=AMP_DTYPE):
                val_loss = _combined_loss(model(Xv), yv).item()

        avg = epoch_loss / len(loader)
        history["loss"].append(avg)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, history


# =============================================================================
# INFERENCE HELPERS
# =============================================================================
def predict_next_day(
    model,
    last_sequence : np.ndarray,
    scaler_y,
    current_ohlc  : np.ndarray,
) -> np.ndarray:
    """
    Predict next-day OHLC.
    next_price = current_ohlc * exp(inverse_transform(model_output))
    """
    model.eval()
    with torch.no_grad():
        x_t  = torch.tensor(last_sequence, dtype=torch.float32).to(DEVICE)
        pred = model(x_t).cpu().numpy()
    log_ret   = scaler_y.inverse_transform(pred)[0]
    next_ohlc = np.array(current_ohlc, dtype=np.float64) * np.exp(log_ret)
    return next_ohlc.astype(np.float32)


def reconstruct_prices_from_returns(
    scaler_y,
    scaled_returns : np.ndarray,
    raw_ohlc_array : np.ndarray,
) -> np.ndarray:
    """
    Batch reconstruct OHLC prices from scaled log-returns.
    Uses previous-step OHLC as baseline.
    """
    log_returns   = scaler_y.inverse_transform(scaled_returns)
    prev_ohlc     = np.concatenate([raw_ohlc_array[:1], raw_ohlc_array[:-1]], axis=0)
    reconstructed = prev_ohlc * np.exp(log_returns)
    return reconstructed.astype(np.float32)