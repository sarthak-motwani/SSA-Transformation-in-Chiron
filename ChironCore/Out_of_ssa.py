#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Out-of-SSA Transformation for ChironLang
"""

from typing import List, Tuple, Dict, Set
from ChironAST.ChironAST import (
    Instruction, PhiCommand, AssignmentCommand, ConditionCommand, BoolExpr, BoolFalse, Var, Num
)
from irhandler import IRHandler
from cfg.cfgBuilder import buildCFG, dumpCFG
from cfg.ChironCFG import ChironCFG, BasicBlock
import bisect

class OutOfSSATransformer:
    def __init__(self, ir: List, cfg: ChironCFG):
        self.ir = ir
        self.cfg = cfg
        self.ir_handler = IRHandler()
        self.split_blocks = {}  # Maps critical edges to new blocks

    def find_all_preds_list(self, preds_list):
        for bb in list(self.cfg.nodes()):
            preds_list[bb] = list(self.cfg.predecessors(bb))
    
    def find_critical_edges(self) -> List[Tuple[BasicBlock, BasicBlock]]:
        """Identify all critical edges in the CFG"""
        critical_edges = []
        for u, v in self.cfg.edges():
            if self.cfg.out_degree(u) > 1 and self.cfg.in_degree(v) > 1:
                critical_edges.append((u, v))
        return critical_edges

    def split_critical_edge(self, u: BasicBlock, v: BasicBlock, preds_list):
        """Insert a new block between u and v to split a critical edge"""
        # Create new block
        n = len(self.cfg.nodes())
        n+=1
        new_block = BasicBlock(n)
        self.cfg.add_node(new_block)
        false_cmd = ConditionCommand(BoolFalse())
        new_block.instrlist.append((false_cmd, 0))
        # Redirect u -> v to u -> new_block -> v
        old_idx_u = -1
        for idx, old_pred in enumerate(preds_list[v]):
            if (old_pred==u):
                old_idx_u = idx

        label = self.cfg.get_edge_label(u, v)
        self.cfg.nxgraph.remove_edge(u, v)
        self.cfg.add_edge(u, new_block, label=label)
        self.cfg.add_edge(new_block, v, label='Cond_False')
        preds_list[v][old_idx_u] = new_block
        
        # Update IR with new block (empty for now)
        # Insert jump from new_block to v (handled during CFG rebuild)
        self.split_blocks[(u, v)] = new_block

    def split_all_critical_edges(self, preds_list):
        """Split all critical edges in the CFG"""
        critical_edges = self.find_critical_edges()
        for u, v in critical_edges:
            self.split_critical_edge(u, v, preds_list)

    def replace_phi_with_copies(self, preds_list):
        """Replace φ-functions with copy instructions in predecessors"""

        for bb in list(self.cfg.nodes()):
            # Extract φ-functions and other instructions
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
                predecessors = preds_list[bb]
                
                if len(operands) != len(predecessors):
                    raise ValueError("Phi operand count mismatch")
                
                for idx, pred in enumerate(predecessors):
                    operand = operands[idx]
                    if(operand.endswith("_0")):
                        temp_assignment = AssignmentCommand(Var(operand), Num(-1))
                        if(isinstance(pred.instrlist[-1][0], ConditionCommand)):
                            pred.instrlist.insert(len(pred.instrlist) - 1, (temp_assignment, insert_pos))
                        else:
                            pred.instrlist.append((temp_assignment, insert_pos))
                    # Create assignment: var = operand
                    assignment = AssignmentCommand(Var(var), Var(operand))
                    insert_pos = bb.instrlist[0][1]
                    if(isinstance(pred.instrlist[-1][0], ConditionCommand)):
                        pred.instrlist.insert(len(pred.instrlist) - 1, (assignment, insert_pos))
                    else:
                        pred.instrlist.append((assignment, insert_pos))
        

    def handle_preds_of_end(self):
        for node in self.cfg.nodes():
            if node.name=='END':
                for preds in self.cfg.predecessors(node):
                    if self.cfg.get_edge_label(preds, node) != 'Cond_False':
                        preds.instrlist.append((ConditionCommand(BoolFalse()), 1))
                break

    def instructions_in_original_order(self, initial_node_order):
        """
        Traverses the CFG and returns a list of instructions in the same order as the original IR.
        """
        # Collect all (instruction, index) tuples from all basic blocks except START and END
        idx_in_ir = 0
        for i, node in enumerate(initial_node_order):
            initial_node_order[i] = node[1]
        for node in self.cfg.nodes():
            if(node not in initial_node_order):
                initial_node_order.append(node)
        for node in initial_node_order:
            if(len(node.instrlist)==0):
                continue
            if len(node.instrlist)==1 and node.instrlist[0][0]==ConditionCommand(BoolFalse()):
                continue
            for i, (instr, _) in enumerate(node.instrlist):
                if(idx_in_ir < len (self.ir)):
                    self.ir[idx_in_ir] = (instr, 1)
                else:
                    self.ir.append((instr, 1))
                node.instrlist[i] = (node.instrlist[i][0], idx_in_ir)
                idx_in_ir+=1


    def Update_Jump_Offsets(self):
        """Update Jump Offsets in IR after inserting instructions"""
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
        
    def transform(self) -> Tuple[List, ChironCFG]:
        """Execute out-of-SSA transformation pipeline"""
        preds_list={} #dict: v, list of predecessors in original order, while splitting stores new preds
        initial_node_order=[]
        for node in self.cfg.nodes():
            if(len(node.instrlist)==0): continue
            initial_node_order.append((node.instrlist[0][1], node))
        
        initial_node_order.sort(key=lambda x: x[0])
        self.find_all_preds_list(preds_list)
        self.split_all_critical_edges(preds_list)
        self.replace_phi_with_copies(preds_list)
        self.handle_preds_of_end()
        self.instructions_in_original_order(initial_node_order)
        self.Update_Jump_Offsets()
        # print(all_instr)
        return self.ir, self.cfg

# ======================== Interface ========================
def out_of_ssa(ir, cfg: ChironCFG) -> Tuple[List, ChironCFG]:
    """Top-level out-of-SSA transformation entry point"""
    transformer = OutOfSSATransformer(ir, cfg)
    transformer.transform()
    dumpCFG(cfg, "old_out_of_ssa")
    new_cfg = buildCFG(ir, 'out_of_ssa2')
    dumpCFG(new_cfg, 'out_of_ssa2')