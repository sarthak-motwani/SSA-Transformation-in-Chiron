#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Out-of-SSA Transformation for ChironLang"""

from typing import List, Tuple, Dict, Set
from ChironAST.ChironAST import (
    Instruction, PhiCommand, AssignmentCommand, ConditionCommand, BoolExpr, BoolFalse, Var, Num
)
from irhandler import IRHandler
from cfg.cfgBuilder import buildCFG, dumpCFG
from cfg.ChironCFG import ChironCFG, BasicBlock
from ssa.SSCP import LatticeValue, SSCP, SSCPOptimizer

# Class to handle all functions for Out of SSA Transformation
class OutOfSSATransformer:
    def __init__(self, ir: List, cfg: ChironCFG, sscp_results:  Dict[str, LatticeValue] = None):
        self.ir = ir
        self.cfg = cfg
        self.ir_handler = IRHandler()
        self.split_blocks = {}  # Maps critical edges to new blocks
        self.preds_list={} #dict: v, list of predecessors in original order, while splitting stores new preds
        self.sscp_results = sscp_results
        self.initial_node_order = self.compute_initial_node_order()

    # Function to convert preds set in nx.graph to a list to retain indices
    def find_all_preds_list(self):
        for bb in list(self.cfg.nodes()):
            self.preds_list[bb] = list(self.cfg.predecessors(bb))
    
    def compute_initial_node_order(self):
        initial_node_order=[]
        for node in self.cfg.nodes():
            if(len(node.instrlist)==0): continue
            initial_node_order.append((node.instrlist[0][1], node))
        initial_node_order.sort(key=lambda x: x[0])
        for i, node in enumerate(initial_node_order):
            initial_node_order[i] = node[1]
        return initial_node_order

    # Identify all critical edges in the CFG
    def find_critical_edges(self) -> List[Tuple[BasicBlock, BasicBlock]]:
        critical_edges = []
        for u, v in self.cfg.edges():
            if self.cfg.out_degree(u) > 1 and self.cfg.in_degree(v) > 1:
                critical_edges.append((u, v))
        return critical_edges

    # Insert a new block between u and v to split a critical edge
    def split_critical_edge(self, u: BasicBlock, v: BasicBlock):
        # Create new block
        n = len(self.cfg.nodes())
        n+=1
        new_block = BasicBlock(n)
        self.cfg.add_node(new_block)
        false_cmd = ConditionCommand(BoolFalse())
        new_block.instrlist.append((false_cmd, 0))
        # Redirect u -> v to u -> new_block -> v
        old_idx_u = -1
        for idx, old_pred in enumerate(self.preds_list[v]):
            if (old_pred==u):
                old_idx_u = idx

        label = self.cfg.get_edge_label(u, v)
        self.cfg.nxgraph.remove_edge(u, v)
        self.cfg.add_edge(u, new_block, label=label)
        self.cfg.add_edge(new_block, v, label='Cond_False')
        self.preds_list[v][old_idx_u] = new_block
        self.split_blocks[(u, v)] = new_block
        if(label=='Cond_True'):
            u_index = self.initial_node_order.index(u)
            self.initial_node_order.insert(u_index, new_block)

    # Split all critical edges in the CFG
    def split_all_critical_edges(self):
        critical_edges = self.find_critical_edges()
        for u, v in critical_edges:
            self.split_critical_edge(u, v)

    # Replace φ-functions with copy instructions in predecessors
    def replace_phi_with_copies(self):
        for bb in list(self.cfg.nodes()):
            phis = []
            other_instrs = []
            for idx, (instr, offset) in enumerate(bb.instrlist):
                if isinstance(instr, PhiCommand):
                    phis.append((instr, offset))
                else:
                    other_instrs.append((instr, offset))
            
            # Remove φ-functions from current block
            bb.instrlist = other_instrs
            
            # Process each φ-function
            for phi, offset in phis:
                var = phi.var
                operands = phi.operands
                predecessors = self.preds_list[bb]
                
                if len(operands) != len(predecessors):
                    raise ValueError("Phi operand count mismatch")
                
                for idx, pred in enumerate(predecessors):
                    operand = operands[idx]
                    if(operand.endswith("_0")):
                        temp_assignment = AssignmentCommand(Var(var), Num(10))
                        if(isinstance(pred.instrlist[-1][0], ConditionCommand)):
                            pred.instrlist.insert(len(pred.instrlist) - 1, (temp_assignment, 0))
                        else:
                            pred.instrlist.append((temp_assignment, 0))
                    # Create assignment: var = operand
                    else:
                        lhs_assgn = Var(operand)
                        if(self.sscp_results!=None and operand in self.sscp_results and self.sscp_results[operand].is_constant()):
                            lhs_assgn = Num(self.sscp_results[operand].get_constant())

                        assignment = AssignmentCommand(Var(var), lhs_assgn)
                        if(isinstance(pred.instrlist[-1][0], ConditionCommand)):
                            pred.instrlist.insert(len(pred.instrlist) - 1, (assignment, 0))
                        else:
                            pred.instrlist.append((assignment, 0))
        
    # Function to specially handle predecessors of 'END' block
    def handle_preds_of_end(self):
        for node in self.cfg.nodes():
            if node.name=='END':
                for preds in self.cfg.predecessors(node):
                    if self.cfg.get_edge_label(preds, node) != 'Cond_False':
                        preds.instrlist.append((ConditionCommand(BoolFalse()), 1))
                break

    # Traverses the CFG and returns a list of instructions in the same order as the original IR
    def instructions_in_original_order(self):
        idx_in_ir = 0
        for node in self.cfg.nodes():
            if(node not in self.initial_node_order):
                if(len(node.instrlist)==0):
                    continue
                self.initial_node_order.append(node)
        
        for node in self.initial_node_order:
            for i, (instr, _) in enumerate(node.instrlist):
                if(idx_in_ir < len (self.ir)):
                    self.ir[idx_in_ir] = (instr, 1)
                else:
                    self.ir.append((instr, 1))
                node.instrlist[i] = (node.instrlist[i][0], idx_in_ir)
                idx_in_ir+=1

    # Update Jump Offsets in IR after inserting instructions
    def Update_Jump_Offsets(self):
        for bb in self.cfg.nodes():        
            if(len(bb.instrlist)==0 and bb.name!='END'):
                continue
            if(bb.name=='END'):
                for pred_block in self.cfg.predecessors(bb):
                    last_instr_idx_pred = pred_block.instrlist[-1][1]
                    self.ir[last_instr_idx_pred] = (self.ir[last_instr_idx_pred][0], len(self.ir) - last_instr_idx_pred)
            else: 
                first_instr_idx_bb = bb.instrlist[0][1]
                for pred_block in self.cfg.predecessors(bb):
                    # last_instr_pred = pred_block.instrlist[-1][0]
                    last_instr_idx_pred = pred_block.instrlist[-1][1]
                    if self.cfg.get_edge_label(pred_block, bb) == 'Cond_False':
                        if(last_instr_idx_pred+self.ir[last_instr_idx_pred][1]!=first_instr_idx_bb):
                            self.ir[last_instr_idx_pred] = (self.ir[last_instr_idx_pred][0], first_instr_idx_bb - last_instr_idx_pred)
        
    # executing out of ssa pipeline
    def transform(self) -> Tuple[List, ChironCFG]: 
        self.find_all_preds_list()
        self.split_all_critical_edges()
        self.replace_phi_with_copies()
        self.handle_preds_of_end()
        self.instructions_in_original_order()
        self.Update_Jump_Offsets()
        return self.ir, self.cfg

# Interface to perform out of ssa transformation
def out_of_ssa(ir, cfg: ChironCFG, sscp_results:  Dict[str, LatticeValue]) -> Tuple[List, ChironCFG]:
    transformer = OutOfSSATransformer(ir, cfg, sscp_results)
    transformer.transform()
    dumpCFG(cfg, "cfg4_old_out_of_ssa")
    new_cfg = buildCFG(ir, 'out_of_ssa2')
    dumpCFG(new_cfg, 'cfg5_new_out_of_ssa')