(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* What the array value leaves as it was.

   An immutable array of values may be a value that a primitive makes
   as the term runs: [Value_immutable_array] (FlambdaValueSpec.v),
   called an array value here. One primitive makes it,
   [Make_immutable_array] (FlambdaPrimitiveSpec.v), and nothing else
   does. A static immutable array is not an array value: it is what a
   symbol is bound to, as it was.

   With the array value two relations between values relate more pairs
   than they did. That of the machine with a heap
   (FlambdaHeapMachineSpec.value_related) does by two rules whose text
   is as it was: [Step_array], through the observation
   FlambdaValueSpec.array_of, which reads an array value; and
   [Step_poison], through FlambdaValueSpec.kind_of_value, whose text is
   as it was too and whose last arm answers [Value] of an array value,
   so that where poison is refined a poison of that kind is refined by
   one. That of the lifting (FlambdaLiftSpec.value_lifted) does by its
   rules [Lifted_array] and [Lifted_to_array]. Statements that were
   proved before the array value
   have one or the other in their conclusions, and some in their
   premises as well. This file says that, of a term that makes no
   array and of memories that hold no array value, those statements
   say what they said.

   This file is to be reviewed: the predicates and every statement.

   - What has no array. [no_array_value] of a value: no array value in
     it, all the way down. Of what a symbol is bound to and of a cell
     of the heap: none in the values that it holds; so the elements of
     a static immutable array hold none. [makes_array] of a term: the
     primitive is somewhere in it, in a handler or in the code that it
     binds included. A state is free of arrays
     ([state_free_of_arrays]) when nothing that it holds has an array
     value (its variables, the environments and the arguments that its
     continuations keep, what its symbols are bound to, its heap) and
     nothing that it may come to run makes one (its expression, the
     handlers that its continuations keep, its code).

   - Invariance. A step keeps a state free of arrays, and what the
     state leaves with holds no array value: for the machine
     ([step_free_of_arrays_statement]) and for the machine with a heap
     ([heap_step_free_of_arrays_statement]). The premise is about the
     whole state and not about its expression alone: a call runs code
     of the static environment, and a continuation a handler that it
     keeps. Underneath is a law of the primitives: one that is not
     [Make_immutable_array] makes no array value out of arguments and
     static data that hold none ([eval_prim_no_array_value_statement],
     and [exec_prim_no_array_value_statement] with a heap). Renaming
     the variables and the continuations of a state neither makes it
     free of arrays nor stops it being so
     ([state_free_of_arrays_renamed_statement]). And the refinement of
     expressions restricted to the closed whole source states that
     are free of arrays (the source expression with its environments
     and its memory) commutes with renaming
     ([heap_expr_refines_free_of_arrays_equivariant_statement]): the
     statement of that for all states (FlambdaHeapEquivarianceSpec.v)
     does not give it, as its premise is about more of them. Nor does
     FlambdaHeapMachineSpec.delete_write_wrong_statement give what it
     gave: it is an existence, of a term whose [Let] cannot be removed
     and, within, of a state that shows it, and either may now have
     arrays. [delete_write_wrong_free_of_arrays_statement] is that
     statement for the refinement so restricted: the term makes no
     array and the state is free of arrays.

   - The observation. Of a value that is not an array value,
     [array_of] answers exactly as it did: for a symbol bound to a
     static array, with its elements, and for nothing else
     ([array_of_no_array_value_statement]). No other observation was
     changed in its text. Those that end with a wildcard answer of an
     array value what the wildcard says: [kind_of_value] answers
     [Value], and the observations of blocks, of closures, of boxed
     numbers, of strings and of partial applications have no answer.

   - The relation of the machine with a heap. Its rules are as they
     were written; the one that reads [array_of] is [Step_array], and
     [Step_poison] reads [kind_of_value].
     Between values that hold no array value, under static
     environments that hold none, a rule relates two values through
     values that hold none ([value_step_no_array_value_statement]).
     Between memories that hold none, a correspondence that is valid
     stays valid when it is cut down to the values that hold none, and
     relates the same such values
     ([value_related_restricted_statement]). So two outcomes that hold
     no array value are related exactly when a correspondence between
     values that hold none relates them
     ([outcome_related_no_array_value_statement]): no value that the
     derivation meets is an array value, [Step_array] is applied to
     symbols bound to static arrays only, and every rule relates what
     it related. No copy of the relation as it was is kept to say
     so: what is stated is that the derivation stays among the values
     that there were.

   - The relation of the lifting. Its rules are as they were written
     but for [Lifted_array] and [Lifted_to_array]. An array value is
     lifted to an array value of its elements lifted, or to a symbol
     bound to an immutable array of its elements lifted, and to
     nothing else ([value_lifted_array_statement]); nothing else is
     lifted to an array value ([value_lifted_to_array_statement]); and
     a value that holds no array value is lifted to values that hold
     none ([value_lifted_no_array_value_statement]; the converse
     fails, as a symbol holds none whatever it is bound to). The rules
     go by the head of the first value and relate its parts, and the
     two rules of the array have an array value for their first
     value: so a derivation from a value that holds no array value
     has neither. That is stated with the rules that have no array
     named ([value_lifted_no_array]: the rules of the relation but
     those two, each as it is, under the same static environment).
     They relate exactly what the relation relates of a first value
     that holds no array value
     ([value_lifted_no_array_exact_statement]). That is of the values
     that hold no array value: it is not said that the relation was
     ever those rules, and it had [Lifted_array] before it had
     [Lifted_to_array]. So of a unit that makes no array,
     FlambdaLiftSpec.lift_simulates_statement gives what it gave: the
     values at its exit hold no array value, and those of the unit
     lifted are those values lifted by the rules that have no array,
     under the static environment of the last state; and the heaps of
     the two exits are alike by those rules
     ([lift_simulates_no_array_statement]).

     Three statements say less, with no rule named. [Lifted_to_array]
     is the one rule that reads a symbol bound to an immutable array,
     so under a static environment with those bindings left out
     ([without_static_arrays]) it relates nothing
     ([no_array_lifted_to_symbol_statement]). Two values that are
     related, the first of which holds no array value, are related
     under it ([value_lifted_reads_no_static_array_statement]), and so
     are the values at the two exits of a unit that makes no array,
     and what the cells of their heaps hold
     ([lift_simulates_free_of_arrays_statement]). The converse is not
     claimed, and the relation under that environment is not the
     relation without the rule: leaving a binding out may uncover
     another binding of the same symbol.

   Nothing here is a premise of another file. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaEquivarianceSpec FlambdaNormaliseSpec FlambdaHeapSpec
  FlambdaHeapMachineSpec FlambdaHeapEquivarianceSpec FlambdaLiftSpec
  FlambdaWellFormedSpec FlambdaMyClosureSpec FlambdaClosuresSpec.
Import ListNotations.

(* ======================================================================== *)
(* What holds no array value                                                *)
(* ======================================================================== *)

Definition is_array_value (v : value) : bool :=
  match v with
  | Value_immutable_array _ => true
  | _ => false
  end.

(* No array value in the value, all the way down. *)
Fixpoint no_array_value (v : value) : bool :=
  match v with
  | Value_immutable_array _ => false
  | Value_const _ | Value_symbol _ | Value_boxed_number _
  | Value_location _ => true
  | Value_closure _ slots | Value_closure_of_set _ _ slots =>
    forallb (fun slot => no_array_value (snd slot)) slots
  | Value_block _ fields => forallb no_array_value fields
  | Value_partial_application callee args _ =>
    no_array_value callee && forallb no_array_value args
  end.

(* None in what a symbol is bound to: in the values of its value slots,
   in its fields, in its contents, in its elements. *)
Definition static_no_array_value (a : statically_allocated_value) : bool :=
  forallb no_array_value (values_of_static a).

(* None in a cell of the heap. *)
Definition cell_no_array_value (c : cell) : bool :=
  forallb no_array_value (values_of_cell c).

(* None in an environment, in a static environment, in a heap; the
   bindings that are shadowed included. *)
Definition env_no_array_value (vars : env) : Prop :=
  ∀ (x : variable) (v : value), (x, v) ∈ vars -> no_array_value v = true.

Definition statics_no_array_value (statics : static_env) : Prop :=
  ∀ (s : symbol) (a : statically_allocated_value),
    (s, a) ∈ static_values statics -> static_no_array_value a = true.

Definition heap_no_array_value (h : heap) : Prop :=
  ∀ (l : location) (c : cell), (l, c) ∈ h -> cell_no_array_value c = true.

(* ======================================================================== *)
(* What makes no array                                                      *)
(* ======================================================================== *)

Definition prim_makes_array (p : prim) : bool :=
  match p with
  | Variadic (Make_immutable_array _) => true
  | _ => false
  end.

Definition named_makes_array (d : named) : bool :=
  match d with
  | Prim p _ => prim_makes_array p
  | _ => false
  end.

(* Whether the primitive is somewhere in the expression: in a body, in
   a handler, in the code that a [Let_static] binds. *)
Fixpoint makes_array (e : expr) : bool :=
  match e with
  | Let _ _ d body => named_makes_array d || makes_array body
  | Let_static bindings body =>
    existsb binding_makes_array bindings || makes_array body
  | Let_cont_non_recursive _ h body =>
    handler_makes_array h || makes_array body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_makes_array h) handlers
    || makes_array body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_makes_array (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => makes_array handler
  end
with binding_makes_array (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_makes_array c
  | Bind_deleted_code _ | Bind_symbol _ _ => false
  end
with code_makes_array (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => makes_array body
  end.

(* ======================================================================== *)
(* A state free of arrays                                                   *)
(* ======================================================================== *)

(* What a continuation stands for is free of arrays: a way out is; a
   handler with the environments of its definition is when the handler
   and the handlers defined with it make none, the values of its
   variables hold none, and what its continuations stand for is free of
   them; a continuation with extra arguments, and what is left of an
   over-application, when the arguments hold none and the
   continuations inside are free of them. *)
Inductive cont_free_of_arrays : cont_value -> Prop :=
  | Outer_free_of_arrays (k : continuation) :
    cont_free_of_arrays (Outer k)
  | Extra_args_free_of_arrays (c : cont_value) (extra : list value) :
    cont_free_of_arrays c ->
    forallb no_array_value extra = true ->
    cont_free_of_arrays (Extra_args c extra)
  | Closure_free_of_arrays (h : continuation_handler) (vars : env)
      (conts : list (continuation * cont_value))
      (recursive : list (continuation * continuation_handler)) :
    handler_makes_array h = false ->
    env_no_array_value vars ->
    (∀ (k : continuation) (c : cont_value),
      (k, c) ∈ conts -> cont_free_of_arrays c) ->
    (∀ (k : continuation) (h' : continuation_handler),
      (k, h') ∈ recursive -> handler_makes_array h' = false) ->
    cont_free_of_arrays (Closure h vars conts recursive)
  | Over_application_free_of_arrays (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    forallb no_array_value remaining = true ->
    cont_free_of_arrays return_to ->
    cont_free_of_arrays exn_to ->
    cont_free_of_arrays
      (Over_application remaining remaining_arity return_arity return_to
         exn_to).

Definition conts_free_of_arrays (conts : cont_env) : Prop :=
  ∀ (k : continuation) (c : cont_value),
    (k, c) ∈ conts -> cont_free_of_arrays c.

(* No code that is bound makes an array. *)
Definition code_makes_no_array (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    (id, c) ∈ static_code statics -> code_makes_array c = false.

Definition state_free_of_arrays (s : state) : Prop :=
  makes_array (state_expr s) = false
  ∧ env_no_array_value (state_vars s)
  ∧ conts_free_of_arrays (state_conts s)
  ∧ code_makes_no_array (state_statics s)
  ∧ statics_no_array_value (state_statics s)
  ∧ heap_no_array_value (state_heap s).

(* The same of a state of the machine with a heap. *)
Definition heap_state_free_of_arrays (s : heap_state) : Prop :=
  state_free_of_arrays (forget_heap s).

(* What a state of that machine leaves with holds no array value: the
   values, and the memory they point into. *)
Definition outcome_no_array_value (o : outcome) : Prop :=
  (∀ v : value, v ∈ outcome_values o -> no_array_value v = true)
  ∧ statics_no_array_value (outcome_statics o)
  ∧ heap_no_array_value (outcome_heap o).

(* ======================================================================== *)
(* Invariance                                                               *)
(* ======================================================================== *)

(* A primitive that is not the one that makes an array makes no array
   value out of none. *)
Definition eval_prim_no_array_value_statement : Prop :=
  ∀ (statics : static_env) (p : prim) (vs : list value) (v : value),
    prim_makes_array p = false ->
    statics_no_array_value statics ->
    (∀ arg : value, arg ∈ vs -> no_array_value arg = true) ->
    eval_prim statics p vs = Some v ->
    no_array_value v = true.

(* And with a heap: its value holds none, and nor does the heap that it
   leaves. *)
Definition exec_prim_no_array_value_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    prim_makes_array p = false ->
    statics_no_array_value statics ->
    heap_no_array_value h ->
    (∀ arg : value, arg ∈ vs -> no_array_value arg = true) ->
    exec_prim statics h p vs = Some (v, h') ->
    no_array_value v = true ∧ heap_no_array_value h'.

(* A step keeps a state free of arrays, and a state that is leaves
   with values and a heap that hold no array value. *)
Definition step_free_of_arrays_statement : Prop :=
  ∀ s : state,
    state_free_of_arrays s ->
    match step s with
    | Step s' => state_free_of_arrays s'
    | Exit _ vs hp rd =>
      (∀ v : value, v ∈ vs -> no_array_value v = true)
      ∧ heap_no_array_value hp
    | Undefined => True
    end.

(* So does a run. A unit that makes no array, run on its own, starts
   in such a state: [State e [] [] empty_static_env empty_heap] is free
   of arrays when [makes_array e = false]. *)
Definition run_free_of_arrays_statement : Prop :=
  ∀ (n : nat) (s : state),
    state_free_of_arrays s ->
    match run n s with
    | Step s' => state_free_of_arrays s'
    | Exit _ vs hp rd =>
      (∀ v : value, v ∈ vs -> no_array_value v = true)
      ∧ heap_no_array_value hp
    | Undefined => True
    end.

(* The same of the machine with a heap. *)
Definition heap_step_free_of_arrays_statement : Prop :=
  ∀ s : heap_state,
    heap_state_free_of_arrays s ->
    match heap_step s with
    | Heap_step s' => heap_state_free_of_arrays s'
    | Heap_exit _ o => outcome_no_array_value o
    | Heap_undefined => True
    end.

Definition heap_run_free_of_arrays_statement : Prop :=
  ∀ (n : nat) (s : heap_state),
    heap_state_free_of_arrays s ->
    match heap_run n s with
    | Heap_step s' => heap_state_free_of_arrays s'
    | Heap_exit _ o => outcome_no_array_value o
    | Heap_undefined => True
    end.

(* Renaming the variables and the continuations of a term brings the
   primitive neither in nor out, and a value holds no variable and no
   continuation: a state renamed is free of arrays exactly when the
   state is. *)
Definition makes_array_renamed_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    makes_array (rename_expr r e) = makes_array e.

Definition state_free_of_arrays_renamed_statement : Prop :=
  ∀ (r : renaming) (s : state),
    state_free_of_arrays (rename_state r s) <-> state_free_of_arrays s.

Definition heap_state_free_of_arrays_renamed_statement : Prop :=
  ∀ (r : renaming) (s : heap_state),
    heap_state_free_of_arrays (rename_heap_state r s)
    <-> heap_state_free_of_arrays s.

(* FlambdaHeapMachineSpec.heap_expr_refines restricted to the closed
   whole source states that are free of arrays. The condition is on the
   source expression with its environments and its memory, so it has
   [makes_array source = false] in it: of a source that makes an
   array it holds of no state, so such a source is refined by every
   target in this sense, and that tells nothing. The condition says
   nothing of the target. The refinement is [heap_refines] as this
   tree has it. That, for a source and a target in neither of
   which the primitive that makes an array occurs, this is the
   refinement of expressions that there was before the array value
   is no statement of this file: it rests on the statements of this
   file (a step keeps the condition; of a value that is not an array
   value [array_of] answers as it did; outcomes that hold no array
   value are related among the values that hold none) and on a
   reading of what changed in the rules of the machine, which
   status/P1.md has. *)
Definition heap_expr_refines_free_of_arrays (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (h : heap),
    state_closed (Heap_state source vars conts statics h) ->
    heap_state_free_of_arrays (Heap_state source vars conts statics h) ->
    heap_refines (Heap_state source vars conts statics h)
      (Heap_state target vars conts statics h).

(* It commutes with renaming, as [heap_expr_refines] does. *)
Definition heap_expr_refines_free_of_arrays_equivariant_statement : Prop :=
  ∀ (r : renaming) (source target : expr),
    heap_expr_refines_free_of_arrays source target ->
    heap_expr_refines_free_of_arrays (rename_expr r source)
      (rename_expr r target).

(* There is a [Let] of a variable that is not used whose removal is no
   refinement, in this restricted sense: the term makes no array, and
   the removal is refuted under a state that is free of arrays.
   FlambdaHeapMachineSpec.delete_write_wrong_statement, which is of
   [heap_expr_refines], does not say so much: its term may make an
   array, and the state that refutes the refinement may hold an array
   value. *)
Definition delete_write_wrong_free_of_arrays_statement : Prop :=
  ∃ (x : variable) (x_kind : flambda_kind) (p : prim) (args : list simple)
    (body : expr),
    var_free_b x body = false
    ∧ ~ heap_expr_refines_free_of_arrays
          (Let x x_kind (Prim p args) body) body.

(* ======================================================================== *)
(* The observation                                                          *)
(* ======================================================================== *)

(* Of a value that is not an array value, [array_of] answers exactly
   as it did: it has an answer when the value is a symbol bound to a
   static array, and the answer is the elements of that array. *)
Definition array_of_no_array_value_statement : Prop :=
  ∀ (statics : static_env) (v : value) (elements : list value),
    is_array_value v = false ->
    (array_of statics v = Some elements
     <-> ∃ s : symbol,
           v = Value_symbol s
           ∧ lookup s (static_values statics)
             = Some (Static_immutable_array elements)).

(* ======================================================================== *)
(* The relation of the machine with a heap                                  *)
(* ======================================================================== *)

(* A rule that relates two values that hold no array value, under
   static environments that hold none, relates them through values
   that hold none: every pair that the rule hands to [R] is such. *)
Definition value_step_no_array_value_statement : Prop :=
  ∀ (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (R : value -> value -> Prop)
    (v v' : value),
    statics_no_array_value statics ->
    statics_no_array_value statics' ->
    no_array_value v = true ->
    no_array_value v' = true ->
    value_step poison_refines r statics statics' R v v' ->
    value_step poison_refines r statics statics'
      (fun w w' =>
         R w w' ∧ no_array_value w = true ∧ no_array_value w' = true)
      v v'.

(* A correspondence that relates no value in which there is an array
   value. *)
Definition corr_no_array_value (r : correspondence) : Prop :=
  ∀ v v' : value,
    corr_values r v v' ->
    no_array_value v = true ∧ no_array_value v' = true.

(* A correspondence cut down to the values that hold no array value. *)
Definition restrict_to_no_array_value (r : correspondence)
    : correspondence :=
  Correspondence (corr_locations r)
    (fun v v' =>
       corr_values r v v'
       ∧ no_array_value v = true
       ∧ no_array_value v' = true).

(* Between memories that hold no array value, a valid correspondence
   cut down is valid, and it relates the values that hold none exactly
   as the whole correspondence did. *)
Definition value_related_restricted_statement : Prop :=
  ∀ (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (h h' : heap),
    statics_no_array_value statics ->
    statics_no_array_value statics' ->
    heap_no_array_value h ->
    heap_no_array_value h' ->
    correspondence_valid poison_refines r statics statics' h h' ->
    correspondence_valid poison_refines (restrict_to_no_array_value r)
      statics statics' h h'
    ∧ ∀ v v' : value,
        no_array_value v = true ->
        no_array_value v' = true ->
        (value_related poison_refines r statics statics' v v'
         <-> value_related poison_refines (restrict_to_no_array_value r)
               statics statics' v v').

(* So two outcomes that hold no array value are related exactly when a
   correspondence between values that hold none relates them. *)
Definition outcome_related_no_array_value_statement : Prop :=
  ∀ (poison_refines : bool) (o o' : outcome),
    outcome_no_array_value o ->
    outcome_no_array_value o' ->
    (outcome_related poison_refines o o'
     <-> ∃ r : correspondence,
           corr_no_array_value r
           ∧ correspondence_valid poison_refines r (outcome_statics o)
               (outcome_statics o') (outcome_heap o) (outcome_heap o')
           ∧ Forall2
               (value_related poison_refines r (outcome_statics o)
                  (outcome_statics o'))
               (outcome_values o) (outcome_values o')).

(* ======================================================================== *)
(* The relation of the lifting                                              *)
(* ======================================================================== *)

(* An array value is lifted to an array value of its elements lifted,
   or to a symbol bound to an immutable array of its elements lifted,
   and to nothing else. *)
Definition value_lifted_array_statement : Prop :=
  ∀ (statics' : static_env) (elements : list value) (v' : value),
    value_lifted statics' (Value_immutable_array elements) v' ->
    ∃ elements' : list value,
      Forall2 (value_lifted statics') elements elements'
      ∧ (v' = Value_immutable_array elements'
         ∨ ∃ s : symbol,
             v' = Value_symbol s
             ∧ lookup s (static_values statics')
               = Some (Static_immutable_array elements')).

(* Nothing else is lifted to an array value. *)
Definition value_lifted_to_array_statement : Prop :=
  ∀ (statics' : static_env) (v : value) (elements' : list value),
    value_lifted statics' v (Value_immutable_array elements') ->
    ∃ elements : list value,
      v = Value_immutable_array elements
      ∧ Forall2 (value_lifted statics') elements elements'.

(* A value that holds no array value is lifted to values that hold
   none. *)
Definition value_lifted_no_array_value_statement : Prop :=
  ∀ (statics' : static_env) (v v' : value),
    value_lifted statics' v v' ->
    no_array_value v = true ->
    no_array_value v' = true.

(* What a symbol is bound to is an immutable array. *)
Definition is_static_array (a : statically_allocated_value) : bool :=
  match a with
  | Static_immutable_array _ => true
  | _ => false
  end.

(* A static environment with the symbols that are bound to an immutable
   array left out. *)
Definition without_static_arrays (statics : static_env) : static_env :=
  Static_env (static_code statics)
    (filter (fun binding => negb (is_static_array (snd binding)))
       (static_values statics)).

(* Under it no array value is lifted to a symbol: the rule
   [Lifted_to_array] relates nothing there. *)
Definition no_array_lifted_to_symbol_statement : Prop :=
  ∀ (statics' : static_env) (elements : list value) (s : symbol),
    ~ value_lifted (without_static_arrays statics')
        (Value_immutable_array elements) (Value_symbol s).

(* Two values that are related, the first of which holds no array
   value, are related under it: the derivation reads no static array.
   The converse is not claimed: leaving a binding out may uncover
   another binding of the same symbol. *)
Definition value_lifted_reads_no_static_array_statement : Prop :=
  ∀ (statics' : static_env) (v v' : value),
    value_lifted statics' v v' ->
    no_array_value v = true ->
    value_lifted (without_static_arrays statics') v v'.

(* So of a unit that makes no array: the values at the exit of the
   unit hold no array value, and nor do the cells of its heap, and
   those at the exit of the unit lifted are related to them under the
   static environment of the last state with its static arrays left
   out. The heaps of the two exits are alike under that environment
   (FlambdaHeapSpec.heap_alike), as
   FlambdaLiftSpec.lift_simulates_statement has them alike under the
   whole of it. *)
Definition lift_simulates_free_of_arrays_statement : Prop :=
  ∀ (u : unit_interface) (lets : list (path * lift_to))
    (known : list (code_id * symbol)) (e : expr),
    let lifted := bind_my_closures known (lift_expr lets root e) in
    well_formed_unit u lifted = true ->
    only_closures_of known lifted = true ->
    makes_array e = false ->
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves (State e [] [] empty_static_env empty_heap no_reads)
        (Exits k vs hp) ->
      (∀ v : value, v ∈ vs -> no_array_value v = true)
      ∧ heap_no_array_value hp
      ∧ ∃ (n : nat) (last : state) (vs' : list value) (hp' : heap)
          (rd : read_record),
          run n (State lifted [] [] empty_static_env empty_heap no_reads)
            = Step last
          ∧ step last = Exit k vs' hp' rd
          ∧ Forall2
              (value_lifted (without_static_arrays (state_statics last)))
              vs vs'
          ∧ heap_alike
              (value_lifted (without_static_arrays (state_statics last)))
              hp hp'.

(* The rules of FlambdaLiftSpec.value_lifted that have no array: its
   rules but [Lifted_array] and [Lifted_to_array], each as it is
   there, under the same static environment. It is not said that the
   relation was ever these rules (it had [Lifted_array] before it had
   [Lifted_to_array]): the statement that follows says what they are,
   which is the relation at the values that hold no array value. *)
(* (The warning silenced is about induction schemes through the lists.) *)
#[local] Set Warnings "-register-all".
Inductive value_lifted_no_array (statics' : static_env)
    : value -> value -> Prop :=
  | No_array_const (c : const) :
    value_lifted_no_array statics' (Value_const c) (Value_const c)
  | No_array_symbol (s : symbol) :
    value_lifted_no_array statics' (Value_symbol s) (Value_symbol s)
  | No_array_boxed_number (c : const) :
    value_lifted_no_array statics' (Value_boxed_number c)
      (Value_boxed_number c)
  | No_array_block (tag : Z) (fields fields' : list value) :
    Forall2 (value_lifted_no_array statics') fields fields' ->
    value_lifted_no_array statics' (Value_block tag fields)
      (Value_block tag fields')
  | No_array_closure (code : code_id)
      (value_slots value_slots' : list (value_slot * value)) :
    Forall2 (slot_lifted (value_lifted_no_array statics')) value_slots
      value_slots' ->
    value_lifted_no_array statics' (Value_closure code value_slots)
      (Value_closure code value_slots')
  | No_array_closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) :
    Forall2 (slot_lifted (value_lifted_no_array statics')) value_slots
      value_slots' ->
    value_lifted_no_array statics'
      (Value_closure_of_set slot function_decls value_slots)
      (Value_closure_of_set slot function_decls value_slots')
  | No_array_partial_application (callee callee' : value)
      (args args' : list value) (args_arity : arity) :
    value_lifted_no_array statics' callee callee' ->
    Forall2 (value_lifted_no_array statics') args args' ->
    value_lifted_no_array statics'
      (Value_partial_application callee args args_arity)
      (Value_partial_application callee' args' args_arity)
  | No_array_to_boxed_number (c : const) (s : symbol) :
    lookup s (static_values statics')
    = Some (Static_boxed_number (Value_const c)) ->
    value_lifted_no_array statics' (Value_boxed_number c) (Value_symbol s)
  | No_array_to_block (tag : Z) (fields fields' : list value) (s : symbol) :
    lookup s (static_values statics') = Some (Static_block tag fields') ->
    Forall2 (value_lifted_no_array statics') fields fields' ->
    value_lifted_no_array statics' (Value_block tag fields) (Value_symbol s)
  | No_array_to_closure (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) (s : symbol)
      (closure_symbols : list (function_slot * symbol)) :
    lookup slot closure_symbols = Some s ->
    map fst closure_symbols = map fst function_decls ->
    static_set_bound statics' closure_symbols function_decls value_slots' ->
    Forall2 (slot_lifted (value_lifted_no_array statics')) value_slots
      value_slots' ->
    value_lifted_no_array statics'
      (Value_closure_of_set slot function_decls value_slots)
      (Value_symbol s)
  | No_array_to_unnamed_closure (code : code_id)
      (value_slots value_slots' : list (value_slot * value)) (s : symbol) :
    lookup s (static_values statics')
    = Some (Static_closure code value_slots') ->
    Forall2 (slot_lifted (value_lifted_no_array statics')) value_slots
      value_slots' ->
    value_lifted_no_array statics' (Value_closure code value_slots)
      (Value_symbol s)
  | No_array_location (l : location) :
    value_lifted_no_array statics' (Value_location l) (Value_location l).

(* Those rules relate exactly what the relation of the lifting relates
   of a first value that holds no array value, under one static
   environment: nothing is asked of that environment. *)
Definition value_lifted_no_array_exact_statement : Prop :=
  ∀ (statics' : static_env) (v v' : value),
    value_lifted_no_array statics' v v'
    <-> value_lifted statics' v v' ∧ no_array_value v = true.

(* So of a unit that makes no array, the values at the exit of the unit
   lifted are those of the unit lifted by the rules that have no array,
   under the static environment of the last state, and the heaps of the
   two exits are alike by those rules: the first conclusion of
   FlambdaLiftSpec.lift_simulates_statement, with those rules. *)
Definition lift_simulates_no_array_statement : Prop :=
  ∀ (u : unit_interface) (lets : list (path * lift_to))
    (known : list (code_id * symbol)) (e : expr),
    let lifted := bind_my_closures known (lift_expr lets root e) in
    well_formed_unit u lifted = true ->
    only_closures_of known lifted = true ->
    makes_array e = false ->
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves (State e [] [] empty_static_env empty_heap no_reads)
        (Exits k vs hp) ->
      ∃ (n : nat) (last : state) (vs' : list value) (hp' : heap)
        (rd : read_record),
        run n (State lifted [] [] empty_static_env empty_heap no_reads)
          = Step last
        ∧ step last = Exit k vs' hp' rd
        ∧ Forall2 (value_lifted_no_array (state_statics last)) vs vs'
        ∧ heap_alike (value_lifted_no_array (state_statics last)) hp hp'.
