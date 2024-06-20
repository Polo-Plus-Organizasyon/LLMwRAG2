import torch
from mwptoolkit.utils.utils import read_json_data, get_model
from mwptoolkit.data.utils import create_dataset, create_dataloader
from mwptoolkit.config.configuration import Config

config = Config.load_from_pretrained('./trained_model/GTS-mawps-single/fold0')

dataset = create_dataset(config)

dataset.dataset_load()

dataloader = create_dataloader(config)(config, dataset)

model = get_model(config["model"])(config,dataset)

state_dict = torch.load('./trained_model/Saligned-mawps-single/fold0/model.pth', map_location=config['device'])

model_dict = model.state_dict()

model.load_state_dict(model_dict, strict=True)

model.eval()


"""from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer
from mwptoolkit.utils.preprocess_tool.equation_operator import infix_to_postfix

def process_and_run_model(problem, model, dataloader, config, dataset):
    # Convert the problem into the expected dictionary format
    problem_data = {
        "sQuestion": problem,
        "lSolutions": [],
        "template_solutions": [],
        "template_equations": []
    }
    
    # Perform number transfer and equation conversion
    processed_data, _, _, _ = number_transfer(
        problem_data, 
        dataset_name=config['dataset'], 
        task_type=config['task_type'], 
        mask_type=config['mask_type'], 
        min_generate_keep=config['min_generate_keep'], 
        linear_dataset=dataset
    )

    # Tokenize the inputs
    inputs = dataloader.build_batch_for_predict(processed_data)
    
    # Run the model
    with torch.no_grad():
        outputs = model(inputs)
    
    # Detokenize the result equation
    result_equation = dataloader.detokenize(outputs)
    return result_equation

# Run the model and get the result
result = process_and_run_model(problem, model, dataloader, config, dataset)
print("Result Equation:", result)"""
