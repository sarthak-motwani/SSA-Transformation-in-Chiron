#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Complete SSA Transformation Implementation for ChironLang 
"""

from typing import Dict, Set, List, Tuple
import networkx as nx
from ChironAST.ChironAST import (
    Instruction, PhiCommand, Var, AssignmentCommand,
    MoveCommand, GotoCommand, ConditionCommand,
    BinArithOp, BinCondOp, UnaryArithOp, NOT,
    Num, BoolTrue, BoolFalse
)
from irhandler import IRHandler
from cfg.cfgBuilder import dumpCFG
from cfg.ChironCFG import BasicBlock, ChironCFG

# ========================= Helper Functions ========================
def get_used_vars(expr) -> Set[str]:
    """Recursively extract all variables from an expression"""
    if isinstance(expr, Var):
        return {expr.varname}
    elif isinstance(expr, (BinArithOp, BinCondOp)):
        return get_used_vars(expr.lexpr) | get_used_vars(expr.rexpr)
    elif isinstance(expr, (UnaryArithOp, NOT)):
        return get_used_vars(expr.expr)
    elif isinstance(expr, (Num, BoolTrue, BoolFalse)):
        return set()
    return set()

# ======================= Dominance Analysis ========================
def compute_dominators(cfg: ChironCFG) -> Dict[BasicBlock, Set[BasicBlock]]:
    """Compute dominance relationships"""
    dominators = {}
    entry = cfg.entry
    all_nodes = set(cfg.nodes())
    
    # Initialize dominance sets
    for node in cfg.nodes():
        dominators[node] = all_nodes if node != entry else {entry}
    
    changed = True
    while changed:
        changed = False
        for node in cfg.nodes():
            if node == entry:
                continue
                
            # Get predecessors and handle empty case
            preds = list(cfg.predecessors(node))
            if preds:
                # Compute intersection of all predecessors' dominators
                predecessor_doms = [dominators[p] for p in preds]
                new_dom = set.intersection(*predecessor_doms)
            else:
                new_dom = set()
                
            new_dom.add(node)  # Node always dominates itself
            
            if new_dom != dominators[node]:
                dominators[node] = new_dom
                changed = True
                
    return dominators

def compute_dominator_tree(dominators: Dict) -> Dict[BasicBlock, List[BasicBlock]]:
    """Build immediate dominator tree by checking dominance relationships."""
    dom_tree = {n: [] for n in dominators}

    for node in dominators:
        candidates = dominators[node] - {node}
        if not candidates:
            continue  # Skip entry node

        # Find the immediate dominator (IDOM)
        idom = None
        for candidate in candidates:
            # Check if this candidate dominates all other candidates
            if all(other in dominators[candidate] for other in candidates if other != candidate):
                idom = candidate
                break

        # Fallback: pick the first candidate (shouldn't happen in valid CFGs)
        if not idom and candidates:
            idom = next(iter(candidates))

        if idom:
            dom_tree[idom].append(node)

    return dom_tree

def compute_dominance_frontiers(
    cfg: ChironCFG,
    dominators: Dict[BasicBlock, Set[BasicBlock]]
) -> Dict[BasicBlock, Set[BasicBlock]]:
    """Compute dominance frontiers using Cytron's algorithm (fixed IDOM handling)"""
    frontiers = {n: set() for n in cfg.nodes()}
    dom_tree = compute_dominator_tree(dominators)

    for node in cfg.nodes():
        predecessors = list(cfg.predecessors(node))
        if len(predecessors) >= 2:  # Merge point (e.g., Block 10)
            for p in predecessors:
                runner = p
                idom = next((d for d, children in dom_tree.items() if node in children), None)
                while runner != idom and runner is not None:
                    frontiers[runner].add(node)
                    # Move to runner's immediate dominator
                    runner = next((d for d, children in dom_tree.items() if runner in children), None)
    return frontiers

# ===================== Live Variable Analysis ======================
def compute_live_vars(cfg: ChironCFG) -> Tuple[Dict, Dict, Dict, Dict]:
    """Compute live variables using dataflow analysis"""
    live_in = {bb: set() for bb in cfg.nodes()}
    live_out = {bb: set() for bb in cfg.nodes()}
    ue_var = {bb: set() for bb in cfg.nodes()}
    var_kill = {bb: set() for bb in cfg.nodes()}

    # First pass: Compute UEVar and VarKill
    for bb in cfg.nodes():
        for instr, _ in bb.instrlist:
            used = set()
            defined = set()

            if isinstance(instr, AssignmentCommand):
                defined.add(instr.lvar.varname)
                used.update(get_used_vars(instr.rexpr))
            elif isinstance(instr, MoveCommand):
                used.update(get_used_vars(instr.expr))
            elif isinstance(instr, GotoCommand):
                used.update(get_used_vars(instr.xcor))
                used.update(get_used_vars(instr.ycor))
            elif isinstance(instr, ConditionCommand):
                used.update(get_used_vars(instr.cond))

            # Update analysis sets
            for var in used:
                if var not in var_kill[bb]:
                    ue_var[bb].add(var)
            var_kill[bb].update(defined)

    # Second pass: Iterate to fixed point
    changed = True
    while changed:
        changed = False
        for bb in cfg.nodes():
            new_live_out = set().union(*(live_in[s] for s in cfg.successors(bb)))
            new_live_in = ue_var[bb] | (new_live_out - var_kill[bb])
            
            if new_live_in != live_in[bb]:
                live_in[bb] = new_live_in
                live_out[bb] = new_live_out
                changed = True

    return ue_var, var_kill, live_in, live_out

# ======================== SSA Transformer ==========================
class SSATransformer:
    """Main SSA transformation engine"""
    
    def __init__(self, ir, cfg: ChironCFG):
        self.cfg = cfg
        self.ir = ir
        self.dominators = compute_dominators(cfg)
        self.dom_tree = compute_dominator_tree(self.dominators)
        self.df = compute_dominance_frontiers(cfg, self.dominators)
        self.ue_var, self.var_kill, self.live_in, self.live_out = compute_live_vars(cfg)
        self.globals = self._compute_globals()
        self._validate_cfg()

    def _validate_cfg(self):
        """Ensure CFG is properly structured"""
        if not isinstance(self.cfg, ChironCFG):
            raise ValueError("Invalid CFG type")
        if not self.cfg.entry:
            raise ValueError("CFG missing entry node")

    def _compute_globals(self) -> Set[str]:
        """Identify variables live across multiple blocks"""
        return set().union(*self.ue_var.values())

    def insert_phi_functions(self):
        """Insert φ-functions at dominance frontiers"""
        for var in self.globals:
            worklist = [bb for bb in self.cfg.nodes() if any(
                isinstance(instr, AssignmentCommand) and instr.lvar.varname == var
                for instr, _ in bb.instrlist
            )]
            
            while worklist:
                bb = worklist.pop(0)
                for df_node in self.df[bb]:
                    if not self._has_phi_for_var(df_node, var):
                        # Get number of predecessors to size operands
                        num_preds = len(list(self.cfg.predecessors(df_node)))
                        phi = PhiCommand(var, [""] * num_preds)  # Initialize with placeholders
                        idx_in_ir = df_node.instrlist[0][1]
                        df_node.instrlist.insert(0, (phi, idx_in_ir))
                        ir_handler = IRHandler()
                        ir_handler.addInstruction(self.ir, phi, idx_in_ir)
                        if df_node not in worklist:
                            worklist.append(df_node)
        
        dumpCFG(self.cfg, "cfg_after_phi_insertion")
        return self.cfg

    def print_ir_before_rename(self, title: str = ""):
        """Print the current IR state for debugging"""
        print(f"\n========== {title} ==========")
        for bb in self.cfg.nodes():
            print(f"\n--- Block {bb.name} ---")
            for instr, offset in bb.instrlist:
                print(f"{instr} [Offset: {offset}]")

    def print_dominator_tree(self):
        for bb in self.cfg.nodes():
            print(f"Dom_tree({bb.name}) = {[df_node.name for df_node in self.dom_tree[bb]]}")

       
    def _has_phi_for_var(self, bb: BasicBlock, var: str) -> bool:
        """Check if block already has φ-function for variable"""
        return any(
            isinstance(instr, PhiCommand) and instr.var == var
            for instr, _ in bb.instrlist
        )

    def _rename_in_expr(self, expr, stacks: Dict[str, List[str]]):
        """Recursively rename variables in expressions (supports both arithmetic and boolean)"""
        if isinstance(expr, Var):
            if expr.varname in stacks and stacks[expr.varname]:
                expr.varname = stacks[expr.varname][-1]
        elif isinstance(expr, (BinArithOp, BinCondOp)):
            self._rename_in_expr(expr.lexpr, stacks)
            self._rename_in_expr(expr.rexpr, stacks)
        elif isinstance(expr, (UnaryArithOp, NOT)):
            self._rename_in_expr(expr.expr, stacks)


    def rename_variables(self) -> ChironCFG:
        """Perform systematic variable renaming for all variables"""
        # Phase 1: Collect ALL variables in the program
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
                # Handle Phi operands
                if isinstance(instr, PhiCommand):
                    all_vars.update(op for op in instr.operands if op)

        # Phase 2: Initialize renaming infrastructure
        counters = {var: 0 for var in all_vars}
        stacks = {var: [] for var in all_vars}
        
        # Initialize stacks with version 0
        for var in all_vars:
            stacks[var].append(f"{var}_0")
            ssa_to_base[var]=var
            counters[var] = 1  # Next version will be _1

        def new_name(var: str) -> str:
            """Generate new SSA name and track base mapping"""
            version = counters[var]
            ssa_name = f"{var}_{version}"
            ssa_to_base[ssa_name] = var  # Track original base
            counters[var] += 1
            stacks[var].append(ssa_name)
            return ssa_name

        def process_block(bb: BasicBlock):
            defined_vars = set()

            # Process instructions
            for idx, (instr, _) in enumerate(bb.instrlist):
                if isinstance(instr, PhiCommand):
                    # Use entire variable name as base
                    ssa_name = instr.var
                    base_var = ssa_to_base.get(ssa_name, ssa_name)
                    new_var = new_name(base_var)
                    instr.var = new_var
                    defined_vars.add(base_var)
                elif isinstance(instr, AssignmentCommand):
                    original_var = ssa_to_base.get(instr.lvar.varname, instr.lvar.varname)
                    self._rename_in_expr(instr.rexpr, stacks)
                    new_var = new_name(original_var)
                    instr.lvar = Var(new_var)
                    defined_vars.add(original_var)
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
                try:
                    pred_idx = preds.index(bb)
                except ValueError:
                    continue

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
            for var in defined_vars:
                if stacks[var]:
                    stacks[var].pop()

        # Start processing from entry block
        entry_node = next(n for n in self.cfg.nodes() if n.name == 'START')
        process_block(entry_node)
        dumpCFG(self.cfg, "cfg_after_rename")
        return self.cfg
# ======================== Interface ========================
def build_ssa(ir, cfg: ChironCFG) -> ChironCFG:
    """Top-level SSA transformation entry point"""
    transformer = SSATransformer(ir, cfg)
    transformer.insert_phi_functions()
    transformer.rename_variables()
    return transformer.cfg