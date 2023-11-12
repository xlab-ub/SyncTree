import yaml
from models.delaypred import GNN4DelaySlew
from utils import parser_from_dict
from eval.evaldelayslew import *
from dataprocess import makegraphs, makejths

if __name__ == "__main__":
    f = open("./parameters_sync.yaml", encoding="utf-8")
    params = yaml.load(stream=f, Loader=yaml.FullLoader)
    args = parser_from_dict(params)
    learner = GNN4DelaySlew(args)
    
    # Train Model
    learner.train()
    # Analyze Results
    delay_slew_results(args)
    # Compare w/o TC loss
    # if args.loss_type:
    #     comp_TC(args)