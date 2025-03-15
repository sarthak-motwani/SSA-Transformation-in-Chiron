#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Out-of-SSA Transformation for ChironLang
"""

from typing import List, Tuple, Dict, Set
from ChironAST.ChironAST import (
    Instruction, PhiCommand, AssignmentCommand, Var
)
from irhandler import IRHandler
from cfg.cfgBuilder import buildCFG, dumpCFG
from cfg.ChironCFG import ChironCFG, BasicBlock

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
        
        # Redirect u -> v to u -> new_block -> v
        old_idx_u = -1
        for idx, old_pred in enumerate(self.cfg.predecessors(v)):
            if (old_pred==u):
                old_idx_u = idx
 
        self.cfg.nxgraph.remove_edge(u, v)
        self.cfg.add_edge(u, new_block)
        self.cfg.add_edge(new_block, v)
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
                    # Create assignment: var = operand
                    assignment = AssignmentCommand(Var(var), Var(operand))
                    
                    # Insert at the end of predecessor's instructions
                    # Find insertion position (before any terminal jumps)
                    insert_pos = bb.instrlist[-1][1]
                    # self.ir_handler.addInstruction(
                    #     self.ir, assignment, insert_pos
                    # )
                    pred.instrlist.append((assignment, insert_pos))
        
        # Rebuild CFG to reflect changes
        self.cfg = buildCFG(self.ir, "out_of_ssa")

    def transform(self) -> Tuple[List, ChironCFG]:
        """Execute out-of-SSA transformation pipeline"""
        preds_list={} #dict: v, list of predecessors in original order, while splitting stores new preds
        self.find_all_preds_list(preds_list)
        self.split_all_critical_edges(preds_list)
        self.replace_phi_with_copies(preds_list)
        return self.ir, self.cfg

# ======================== Interface ========================
def out_of_ssa(ir, cfg: ChironCFG) -> Tuple[List, ChironCFG]:
    """Top-level out-of-SSA transformation entry point"""
    transformer = OutOfSSATransformer(ir, cfg)
    transformer.transform()
    dumpCFG(cfg, "out_of_ssa")