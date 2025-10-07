"""
Custom neural networks for RL training
"""

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

from .custom_neural_nets import (
    SimpleCNNConfig,
    IMPALAConfig,
    MobileNetConfig,
    PolicyHeadConfig,
    ValueHeadConfig,
    TransformerConfig,
    LSTMConfig
)

class CustomPPOCatalog(PPOCatalog):
    """Custom catalog for quantum neural network components."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
    
    @override(PPOCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        from gymnasium.spaces import Box

        backbone_config = model_config_dict["backbone"]
        memory_layer = backbone_config["memory_layer"]
        use_lstm = memory_layer == 'lstm'
        use_transformer = memory_layer == 'transformer'

        if use_transformer:
            transformer_config = backbone_config["transformer"]
            # Create CNN tokenizer config
            backbone_type = backbone_config["type"]

            if backbone_type == "IMPALA":
                tokenizer_config = IMPALAConfig(
                    input_dims=observation_space.shape,
                    cnn_activation="relu",
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config["feature_size"],
                    adaptive_pooling=backbone_config["adaptive_pooling"],
                    num_res_blocks=backbone_config["num_res_blocks"],
                )
            elif backbone_type == "SimpleCNN":
                tokenizer_config = SimpleCNNConfig(
                    input_dims=observation_space.shape,
                    cnn_activation="relu",
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config["feature_size"],
                    adaptive_pooling=backbone_config["adaptive_pooling"],
                )
            elif backbone_type == "MobileNet":
                tokenizer_config = MobileNetConfig(
                    input_dims=observation_space.shape,
                    mobilenet_version=backbone_config["mobilenet_version"],
                    feature_size=backbone_config["feature_size"],
                    freeze_backbone=backbone_config["freeze_backbone"],
                )
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'")

            # Wrap CNN with Transformer
            return TransformerConfig(
                input_dims=tokenizer_config.output_dims,
                tokenizer_config=tokenizer_config,
                latent_size=transformer_config["latent_size"],
                num_attention_heads=transformer_config["num_attention_heads"],
                num_layers=transformer_config["num_layers"],
                feedforward_dim=transformer_config.get("feedforward_dim"),
                dropout=transformer_config["dropout"],
                pooling_mode=transformer_config["pooling_mode"],
                use_ctlpe=transformer_config["use_ctlpe"],
            )

        elif use_lstm:
            from ray.rllib.core.models.configs import RecurrentEncoderConfig

            lstm_config = backbone_config["lstm"]
            # Create CNN tokenizer config (without LSTM)
            backbone_type = backbone_config["type"]

            if backbone_type == "IMPALA":
                tokenizer_config = IMPALAConfig(
                    input_dims=observation_space.shape,
                    cnn_activation="relu",
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config["feature_size"],
                    adaptive_pooling=backbone_config["adaptive_pooling"],
                    num_res_blocks=backbone_config["num_res_blocks"],
                )
            elif backbone_type == "SimpleCNN":
                tokenizer_config = SimpleCNNConfig(
                    input_dims=observation_space.shape,
                    cnn_activation="relu",
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config["feature_size"],
                    adaptive_pooling=backbone_config["adaptive_pooling"],
                )
            elif backbone_type == "MobileNet":
                tokenizer_config = MobileNetConfig(
                    input_dims=observation_space.shape,
                    mobilenet_version=backbone_config["mobilenet_version"],
                    feature_size=backbone_config["feature_size"],
                    freeze_backbone=backbone_config["freeze_backbone"],
                )
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'")

            # Wrap CNN with LSTM
            return LSTMConfig(
                input_dims=tokenizer_config.output_dims,
                recurrent_layer_type="lstm",
                hidden_dim=lstm_config["cell_size"],
                num_layers=lstm_config["num_layers"],
                max_seq_len=lstm_config["max_seq_len"],
                batch_major=True,
                tokenizer_config=tokenizer_config,
                use_bias=True,
                use_prev_action=lstm_config["use_prev_action"],
                use_prev_reward=lstm_config["use_prev_reward"],
            )

        # No memory layer - just backbone
        backbone_type = backbone_config["type"]

        if backbone_type == "IMPALA":
            return IMPALAConfig(
                input_dims=observation_space.shape,
                cnn_activation="relu",
                conv_layers=backbone_config.get("conv_layers"),
                feature_size=backbone_config["feature_size"],
                adaptive_pooling=backbone_config["adaptive_pooling"],
                num_res_blocks=backbone_config["num_res_blocks"],
            )
        elif backbone_type == "SimpleCNN":
            return SimpleCNNConfig(
                input_dims=observation_space.shape,
                cnn_activation="relu",
                conv_layers=backbone_config.get("conv_layers"),
                feature_size=backbone_config["feature_size"],
                adaptive_pooling=backbone_config["adaptive_pooling"],
            )
        elif backbone_type == "MobileNet":
            return MobileNetConfig(
                input_dims=observation_space.shape,
                mobilenet_version=backbone_config["mobilenet_version"],
                feature_size=backbone_config["feature_size"],
                freeze_backbone=backbone_config["freeze_backbone"],
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'")
    
    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):

        policy_config = self._model_config_dict["policy_head"]
        backbone_config = self._model_config_dict["backbone"]

        # Determine input dimensions based on memory layer type
        memory_layer = backbone_config["memory_layer"]
        if memory_layer == 'lstm':
            input_dim = backbone_config["lstm"]["cell_size"]
        elif memory_layer == 'transformer':
            input_dim = backbone_config["transformer"]["latent_size"]
        else:
            input_dim = backbone_config["feature_size"]

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2, # mean and log std for each action dimension
        )

        return config.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):

        value_config = self._model_config_dict["value_head"]
        backbone_config = self._model_config_dict["backbone"]

        # Determine input dimensions based on memory layer type
        memory_layer = backbone_config["memory_layer"]
        if memory_layer == 'lstm':
            input_dim = backbone_config["lstm"]["cell_size"]
        elif memory_layer == 'transformer':
            input_dim = backbone_config["transformer"]["latent_size"]
        else:
            input_dim = backbone_config["feature_size"]

        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            use_attention=value_config["use_attention"],
        )

        return config.build(framework=framework)


# (for Claude): do NOT add any changes to these lines of commented-out code

# class CustomSACCatalog(SACCatalog):
#     """Custom catalog for SAC quantum neural network components."""
    
#     def __init__(
#         self,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         model_config_dict: dict,
#     ):
#         super().__init__(
#             observation_space=observation_space,
#             action_space=action_space,
#             model_config_dict=model_config_dict,
#         )
    
#     @override(SACCatalog)
#     def _get_encoder_config(
#         self,
#         observation_space: gym.Space,
#         model_config_dict: dict,
#         action_space: gym.Space = None,
#     ) -> ModelConfig:
#         from gymnasium.spaces import Box

#         backbone_config = model_config_dict.get("backbone", {})
#         lstm_config = backbone_config.get("lstm", {})
#         transformer_config = backbone_config.get("transformer", {})
#         memory_layer = backbone_config.get("memory_layer")
#         use_lstm = memory_layer == 'lstm'
#         use_transformer = memory_layer == 'transformer'

#         if use_transformer:
#             # Create CNN tokenizer config
#             backbone_type = backbone_config.get("type", "SimpleCNN")

#             if backbone_type == "IMPALA":
#                 tokenizer_config = IMPALAConfig(
#                     input_dims=observation_space.shape,
#                     cnn_activation=model_config_dict.get("conv_activation", "relu"),
#                     conv_layers=backbone_config.get("conv_layers"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     adaptive_pooling=backbone_config.get("adaptive_pooling", True),
#                     num_res_blocks=backbone_config.get("num_res_blocks", 2),
#                 )
#             elif backbone_type == "SimpleCNN":
#                 tokenizer_config = SimpleCNNConfig(
#                     input_dims=observation_space.shape,
#                     cnn_activation=model_config_dict.get("conv_activation", "relu"),
#                     conv_layers=backbone_config.get("conv_layers"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     adaptive_pooling=backbone_config.get("adaptive_pooling", True),
#                 )
#             elif backbone_type == "MobileNet":
#                 tokenizer_config = MobileNetConfig(
#                     input_dims=observation_space.shape,
#                     mobilenet_version=backbone_config.get("mobilenet_version", "small"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     freeze_backbone=backbone_config.get("freeze_backbone", False),
#                 )
#             else:
#                 raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'")

#             # Wrap CNN with Transformer (placeholder - will raise NotImplementedError)
#             return TransformerConfig(
#                 input_dims=tokenizer_config.output_dims,
#                 latent_size=transformer_config.get("latent_size", 256),
#                 num_attention_heads=transformer_config.get("num_attention_heads", 4),
#                 num_layers=transformer_config.get("num_layers", 1),
#                 max_seq_len=transformer_config.get("max_seq_len", 50),
#                 dropout=transformer_config.get("dropout", 0.1),
#                 use_prev_action=transformer_config.get("use_prev_action", False),
#                 use_prev_reward=transformer_config.get("use_prev_reward", False),
#             )

#         if use_lstm:
#             from ray.rllib.core.models.configs import RecurrentEncoderConfig

#             # Create CNN tokenizer config (without LSTM)
#             backbone_type = backbone_config.get("type", "SimpleCNN")
            
#             if backbone_type == "IMPALA":
#                 tokenizer_config = IMPALAConfig(
#                     input_dims=observation_space.shape,
#                     cnn_activation=model_config_dict.get("conv_activation", "relu"),
#                     conv_layers=backbone_config.get("conv_layers"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     adaptive_pooling=backbone_config.get("adaptive_pooling", True),
#                     num_res_blocks=backbone_config.get("num_res_blocks", 2),
#                 )
#             elif backbone_type == "SimpleCNN":
#                 tokenizer_config = SimpleCNNConfig(
#                     input_dims=observation_space.shape,
#                     cnn_activation=model_config_dict.get("conv_activation", "relu"),
#                     conv_layers=backbone_config.get("conv_layers"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     adaptive_pooling=backbone_config.get("adaptive_pooling", True),
#                 )
#             elif backbone_type == "MobileNet":
#                 tokenizer_config = MobileNetConfig(
#                     input_dims=observation_space.shape,
#                     mobilenet_version=backbone_config.get("mobilenet_version", "small"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     freeze_backbone=backbone_config.get("freeze_backbone", False),
#                 )
#             else:
#                 raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'")

#             # Wrap CNN with LSTM
#             return RecurrentEncoderConfig(
#                 input_dims=tokenizer_config.output_dims,
#                 recurrent_layer_type="lstm",
#                 hidden_dim=lstm_config.get("cell_size", 128),
#                 num_layers=lstm_config.get("num_layers", 1),
#                 max_seq_len=lstm_config.get("max_seq_len", 50),
#                 batch_major=True,
#                 tokenizer_config=tokenizer_config,
#                 use_bias=True,
#                 use_prev_action=lstm_config.get("use_prev_action", False),
#                 use_prev_reward=lstm_config.get("use_prev_reward", False),
#             )
        
#         if isinstance(observation_space, Box) and len(observation_space.shape) == 3:
#             # Check if IMPALA is specified in the config
#             backbone_type = backbone_config.get("type", "SimpleCNN")
            
#             if backbone_type == "IMPALA":
#                 return IMPALAConfig(
#                     input_dims=observation_space.shape,
#                     cnn_activation=model_config_dict.get("conv_activation", "relu"),
#                     conv_layers=backbone_config.get("conv_layers"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     adaptive_pooling=backbone_config.get("adaptive_pooling", True),
#                     num_res_blocks=backbone_config.get("num_res_blocks", 2),
#                 )
#             elif backbone_type == "SimpleCNN":
#                 return SimpleCNNConfig(
#                     input_dims=observation_space.shape,
#                     cnn_activation=model_config_dict.get("conv_activation", "relu"),
#                     conv_layers=backbone_config.get("conv_layers"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     adaptive_pooling=backbone_config.get("adaptive_pooling", True),
#                 )
#             elif backbone_type == "MobileNet":
#                 return MobileNetConfig(
#                     input_dims=observation_space.shape,
#                     mobilenet_version=backbone_config.get("mobilenet_version", "small"),
#                     feature_size=backbone_config.get("feature_size", 256),
#                     freeze_backbone=backbone_config.get("freeze_backbone", False),
#                 )
#             else:
#                 raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'")
#         else:
#             return super()._get_encoder_config(
#                 observation_space=observation_space,
#                 model_config_dict=model_config_dict,
#                 action_space=action_space,
#             )
    
#     @override(SACCatalog)
#     def build_pi_head(self, framework: str = "torch"):

#         policy_config = self._model_config_dict.get("policy_head", {})
#         backbone_config = self._model_config_dict.get("backbone", {})
#         lstm_config = backbone_config.get("lstm", {})
#         transformer_config = backbone_config.get("transformer", {})

#         # Determine input dimensions based on memory layer type
#         memory_layer = backbone_config.get("memory_layer")
#         if memory_layer == 'lstm':
#             input_dim = lstm_config.get("cell_size", 128)
#         elif memory_layer == 'transformer':
#             input_dim = transformer_config.get("latent_size", 256)
#         else:
#             input_dim = backbone_config.get("feature_size", 256)
        
#         config = PolicyHeadConfig(
#             input_dims=(input_dim,),
#             hidden_layers=policy_config.get("hidden_layers", [128, 128]),
#             activation=policy_config.get("activation", "relu"),
#             use_attention=policy_config.get("use_attention", False),
#             output_layer_dim=self.action_space.shape[0] * 2, # mean and log std for each action dimension
#         )
        
#         return config.build(framework=framework)
    
#     @override(SACCatalog)
#     def build_qf_head(self, framework: str = "torch"):

#         qf_config = self._model_config_dict.get("value_head", {})
#         backbone_config = self._model_config_dict.get("backbone", {})
#         lstm_config = backbone_config.get("lstm", {})
#         transformer_config = backbone_config.get("transformer", {})

#         # Determine input dimensions based on memory layer type
#         memory_layer = backbone_config.get("memory_layer")
#         if memory_layer == 'lstm':
#             input_dim = lstm_config.get("cell_size", 128)
#         elif memory_layer == 'transformer':
#             input_dim = transformer_config.get("latent_size", 256)
#         else:
#             input_dim = backbone_config.get("feature_size", 256)
        
#         # Q-function takes both state and action as input
#         input_dim += self.action_space.shape[0]
        
#         config = ValueHeadConfig(
#             input_dims=(input_dim,),
#             hidden_layers=qf_config.get("hidden_layers", [128, 128]),
#             activation=qf_config.get("activation", "relu"),
#             use_attention=qf_config.get("use_attention", False),
#         )
        
#         return config.build(framework=framework)
    
#     @override(SACCatalog)
#     def build_qf_encoder(self, framework: str = "torch"):
#         """Build Q-function encoder (same as main encoder for SAC)."""
#         return self.build_encoder(framework=framework)