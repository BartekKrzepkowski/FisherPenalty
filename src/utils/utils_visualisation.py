ee_tensorboard_layout = lambda params_names: {
    "running_evaluators_per_layer": {
        'running_loss_per_layer(train)':
            ["Multiline", [f'running_traces_per_param/param_{k}/training' for k in params_names]],
        'running_loss_per_layer(test)':
            ["Multiline", [f'running_traces_per_param/param_{k}/test' for k in params_names]],
    },
    "epoch_evaluators_per_layer": {
        'epoch_loss_per_layer (train)':
            ["Multiline", [f'epoch_traces_per_param/param_{k}/training' for k in params_names]],
        'epoch_loss_per_layer (test)':
            ["Multiline", [f'epoch_traces_per_param/param_{k}/test' for k in params_names]],
    },
}
