#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" SSA Transformation Implementation for ChironLang """

from typing import Dict, Set, List, Tuple
import networkx as nx
from ChironAST.ChironAST import (
    Instruction, PhiCommand, Var, AssignmentCommand,
    MoveCommand, GotoCommand, ConditionCommand,
    BinArithOp, BinCondOp, UnaryArithOp, NOT,
    Num, BoolTrue, BoolFalse
)
from cfg.cfgBuilder import dumpCFG, buildCFG
from cfg.ChironCFG import BasicBlock, ChironCFG
from dominanceFrontiers import (compute_dominators, compute_dominator_tree, 
                                compute_dominance_frontiers)
from liveVariables import compute_live_vars
import bisect

# Function to recursively extract used variables in an expression
def get_used_vars(expr) -> Set[str]:
    if isinstance(expr, Var):
        return {expr.varname}
    elif isinstance(expr, (BinArithOp, BinCondOp)):
        return get_used_vars(expr.lexpr) | get_used_vars(expr.rexpr)
    elif isinstance(expr, (UnaryArithOp, NOT)):
        return get_used_vars(expr.expr)
    elif isinstance(expr, (Num, BoolTrue, BoolFalse)):
        return set()
    return set()

# Class for SSA Transformation
class SSATransformer:
    def __init__(self, ir, cfg: ChironCFG):
        self.cfg = cfg
        self.ir = ir
        self.dominators = compute_dominators(cfg)
        self.dom_tree = compute_dominator_tree(self.dominators)
        self.df = compute_dominance_frontiers(cfg, self.dominators)
        self.ue_var, self.var_kill, self.live_in, self.live_out = compute_live_vars(cfg)
        self.globals = self._compute_globals()

    # Function to compute global variables (variables whose liveness span multiple blocks)
    def _compute_globals(self) -> Set[str]:
        return set().union(*self.ue_var.values())

    # Function to insert phi-functions in the IR using positions known from CFG
    def insert_phi_in_ir(self, idx_phi):
        idx_phi.sort(key=lambda x: x[0])
        real_ctr = 0
        for i_p in idx_phi:
            real_idx = i_p[0]+real_ctr
            self.ir.insert(real_idx, (i_p[1], 1))
            real_ctr+=1

    # Helper function to make instruction indices in IR and (old) CFG same 
    # (used in updating jump offsets in IR)
    def synchronize_cfg_ir(self, idx_phi):
        # synchronizing all except phi-instructions
        idx_list = [idx for idx, _ in idx_phi]
        for bb in self.cfg.nodes():
            for i, (instr, idx) in enumerate(bb.instrlist):
                if(isinstance(instr, PhiCommand)):
                    continue
                count = bisect.bisect_right(idx_list, idx)
                bb.instrlist[i] = (bb.instrlist[i][0], idx + count)
        
        # synchronizing the phi-instructions
        for bb in self.cfg.nodes():
            num_phi = 0 # number of phi_instructions in the bb
            for i, (instr, idx) in enumerate(bb.instrlist):
                if(isinstance(instr, PhiCommand)):
                    num_phi+=1
                else:
                    for j in range(num_phi):
                        bb.instrlist[j] = (bb.instrlist[j][0], idx-num_phi+j)
                    break
    
                if(num_phi==0):
                    break

    # Update Jump Offsets in IR after inserting instructions
    def Update_Jump_Offsets(self):
        for bb in self.cfg.nodes():
            if(len(bb.instrlist)==0 and bb.name!='END'):
                continue
            if(bb.name=='END' and len(bb.instrlist)==0):
                for pred_block in self.cfg.predecessors(bb):
                    last_instr_idx_pred = pred_block.instrlist[-1][1]
                    self.ir[last_instr_idx_pred] = (self.ir[last_instr_idx_pred][0], len(self.ir) - last_instr_idx_pred)
            else: 
                first_instr_idx_bb = bb.instrlist[0][1]
                for pred_block in self.cfg.predecessors(bb):
                    last_instr_idx_pred = pred_block.instrlist[-1][1]
                    if self.cfg.get_edge_label(pred_block, bb) == 'Cond_False':
                        if(last_instr_idx_pred+self.ir[last_instr_idx_pred][1]!=first_instr_idx_bb):
                            self.ir[last_instr_idx_pred] = (self.ir[last_instr_idx_pred][0], first_instr_idx_bb - last_instr_idx_pred)

    
    # Function to insert phi-functions for global variables at dom. frontiers
    def insert_phi_functions(self):
        idx_phi=[] # List to keep track of positions to insert in IR
        for var in self.globals:
            worklist = [bb for bb in self.cfg.nodes() if any(
                isinstance(instr, AssignmentCommand) and instr.lvar.varname == var
                for instr, _ in bb.instrlist
            )]
            
            while worklist:
                bb = worklist.pop(0)
                for df_node in self.df[bb]:
                    if not self._has_phi_for_var(df_node, var):
                        num_preds = len(list(self.cfg.predecessors(df_node)))
                        phi = PhiCommand(var, [""] * num_preds)
                        if(len(df_node.instrlist)!=0): idx_in_ir = df_node.instrlist[0][1] 
                        else: idx_in_ir = len(self.ir)
                        df_node.instrlist.insert(0, (phi, idx_in_ir))
                        idx_phi.append((idx_in_ir, phi))
                        if df_node not in worklist:
                            worklist.append(df_node)

        # Adding phi-instructions to the actual IR
        self.insert_phi_in_ir(idx_phi)

        #syncronizing instruction indices in CFG and IR
        self.synchronize_cfg_ir(idx_phi)
        
        #updating jump offsets in original IR
        self.Update_Jump_Offsets()

        dumpCFG(self.cfg, "cfg1_old_after_phi_insertion")
        return self.cfg


    # Function to check if a basic block already has phi-function for a variable  
    def _has_phi_for_var(self, bb: BasicBlock, var: str) -> bool:
        return any(
            isinstance(instr, PhiCommand) and instr.var == var
            for instr, _ in bb.instrlist
        )

    # Function to recursively rename variables in artith. or bool expressions
    def _rename_in_expr(self, expr, stacks: Dict[str, List[str]]):
        if isinstance(expr, Var):
            if expr.varname in stacks and stacks[expr.varname]:
                expr.varname = stacks[expr.varname][-1]
        elif isinstance(expr, (BinArithOp, BinCondOp)):
            self._rename_in_expr(expr.lexpr, stacks)
            self._rename_in_expr(expr.rexpr, stacks)
        elif isinstance(expr, (UnaryArithOp, NOT)):
            self._rename_in_expr(expr.expr, stacks)

    # Function to perform variable renaming for all instructions
    def rename_variables(self) -> ChironCFG:
        all_vars = set()
        ssa_to_base = {}  # Tracks SSA names to original bases

        # First pass: Identify all variables
        for bb in self.cfg.nodes():
            for instr, _ in bb.instrlist:
                # Variables defined
                if isinstance(instr, (AssignmentCommand, PhiCommand)):
                    var_name = instr.lvar.varname if isinstance(instr, AssignmentCommand) else instr.var
                    all_vars.add(var_name)
                # Variables used
                used = set()
                if isinstance(instr, (AssignmentCommand, MoveCommand, GotoCommand, ConditionCommand)):
                    if isinstance(instr, AssignmentCommand):
                        used = get_used_vars(instr.rexpr)
                    elif isinstance(instr, MoveCommand):
                        used = get_used_vars(instr.expr)
                    elif isinstance(instr, GotoCommand):
                        used = get_used_vars(instr.xcor) | get_used_vars(instr.ycor)
                    elif isinstance(instr, ConditionCommand):
                        used = get_used_vars(instr.cond)
                    all_vars.update(used)
                
                if isinstance(instr, PhiCommand):
                    all_vars.update(op for op in instr.operands if op)

        # Pass 2: Initialize renaming 
        counters = {var: 0 for var in all_vars}
        stacks = {var: [] for var in all_vars}
        
        # Initialize stacks with version 0
        for var in all_vars:
            stacks[var].append(f"{var}_0")
            ssa_to_base[var]=var
            counters[var] = 1  # Next version will be _1

        # Generate new SSA name and track base mapping
        def new_name(var: str) -> str:
            version = counters[var]
            ssa_name = f"{var}_{version}"
            ssa_to_base[ssa_name] = var  # Track original base
            counters[var] += 1
            stacks[var].append(ssa_name)
            return ssa_name

        # Process a single block
        def process_block(bb: BasicBlock):
            defined_vars = dict()

            for idx, (instr, _) in enumerate(bb.instrlist):
                if isinstance(instr, PhiCommand):
                    ssa_name = instr.var
                    base_var = ssa_to_base.get(ssa_name, ssa_name)
                    new_var = new_name(base_var)
                    instr.var = new_var
                    defined_vars[base_var] = defined_vars.get(base_var, 0)+1
                elif isinstance(instr, AssignmentCommand):
                    original_var = ssa_to_base.get(instr.lvar.varname, instr.lvar.varname)
                    self._rename_in_expr(instr.rexpr, stacks)
                    new_var = new_name(original_var)
                    instr.lvar = Var(new_var)
                    defined_vars[original_var] = defined_vars.get(original_var, 0) + 1
                elif isinstance(instr, MoveCommand):
                    self._rename_in_expr(instr.expr, stacks)
                elif isinstance(instr, GotoCommand):
                    self._rename_in_expr(instr.xcor, stacks)
                    self._rename_in_expr(instr.ycor, stacks)
                elif isinstance(instr, ConditionCommand):
                    self._rename_in_expr(instr.cond, stacks)

            # Update phi operands in successors
            for succ in self.cfg.successors(bb):
                preds = list(self.cfg.predecessors(succ))
                pred_idx = preds.index(bb)
                
                for phi_instr, _ in succ.instrlist:
                    if isinstance(phi_instr, PhiCommand):
                        # Resolve base variable using SSA map
                        ssa_name = phi_instr.var
                        base_var = ssa_to_base.get(ssa_name, None)
                        phi_instr.operands[pred_idx] = stacks[base_var][-1]

            # Process children in dominator tree
            for child in self.dom_tree.get(bb, []):
                process_block(child)

            # Roll back stacks
            for var in defined_vars.keys():
                while(defined_vars[var]):
                    if stacks[var]:
                        stacks[var].pop()
                    defined_vars[var]-=1

        # Start processing from entry block
        entry_node = next(n for n in self.cfg.nodes() if n.name == 'START')
        process_block(entry_node)
        dumpCFG(self.cfg, "cfg2_old_after_rename")
        return self.cfg


# Interface to perform ssa transformation
def build_ssa(ir, cfg: ChironCFG) -> ChironCFG:
    transformer = SSATransformer(ir, cfg)
    transformer.insert_phi_functions()
    transformer.rename_variables()
    post_ssa_CFG = buildCFG(ir, "post_ssa_control_flow_graph")
    dumpCFG(post_ssa_CFG, "cfg3_new_post_ssa")
    return post_ssa_CFG