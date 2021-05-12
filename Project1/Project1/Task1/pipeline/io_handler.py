

class IOHandler:
    def __init__(self):
        pass

    # TODO: create a class to handle IO
    # TODO: get best validation error and store in best temp file
    # TODO: compare best validation error with overall best validation error and update file if needed
    # TODO: save best model
    """
    - initial setup to create all files
    - update function for loss_running_best_model and loss_best_model
        update_best_running(training_loss, validation_loss, model)
    - update function to compare and evaluate best model overall
        update_best_model()
        read both loss_running_best_model.txt and loss_best_model
        compare them
        update if necessary
    - load stored model
    - save model

    """

    """
    dir: files

    run_dd:mm:yy_hh:mm:ss.txt:
            file for run_conf to write down all parameters and corresponding losses
    loss_running_best_model.txt
        file for run_conf to write down the current best model(param + losses). 
        Can be evaluated during training
    running_best_model.pt
        file to store the best running model
    loss_best_model.txt
        file for main to write down the best model overall (param + losses). 
        Need to compare this file to the loss_running_best_model.txt and update if needed
    best_model.pt
        file to store the best overall model
    """