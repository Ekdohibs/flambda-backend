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

(* What every set of closures that has a given code at a given function
   slot has: a property of a unit that its run keeps.

   This file is to be reviewed: [sets_made], which reads the sets of
   closures that a term makes; [set_fact], what may be known of a set;
   [sets_made_have], the test; [state_sets_have], what the test gives
   of a state; and the statements.

   The code of a closure does not say which set the closure belongs
   to: two sets of a unit may have the same code at the same function
   slot, beside different code, and with different values in their
   value slots. So in the body of code [c], nothing is known of the
   set that [my_closure] is of but that it has [c] at the function slot
   that [my_closure] is at. What is here is what more can be known,
   where a test of the unit passes: that every set that has [c] at the
   function slot [f] has a given code at another function slot, or
   gives a value slot a given simple ([set_fact]). A fact is closed
   where that simple is a symbol or a constant ([fact_closed]; a fact
   about the code of a function slot is closed).

   - The test ([sets_made_have]) is of the sets that the unit makes,
     by a [Let] or statically, in handlers and in code too
     ([sets_made]): each of them that has [c] at [f] has the fact.
     Nothing is guessed and nothing is believed of what is read.
   - Where the test passes, the state that the unit starts in, run on
     its own, has it of every set that it holds, in its terms and in
     its values ([state_sets_have]; [state_sets_have_initial_statement]),
     and a step keeps it where the fact is closed
     ([state_sets_have_step_statement], which is
     FlambdaContextWhereSpec.kept_by_step of it under the premise
     [fact_closed]). So every state of the run has a closed fact that
     the test passes. Under other environments this fails: a closure
     from elsewhere may have [c] at [f] beside other code. A fact
     about a variable can be written, and the laws of the state that
     the unit starts in and of a closure looked up do not ask that
     the fact is closed; the law of a step does, as the value of a
     variable is computed when the unit runs.
   - In a state that has it, a closure that a variable is bound to,
     at [f] of a set that has [c] there, is of a set that has the fact
     ([closure_set_has_statement]): the code of the other slot, so that
     the projection of that function slot from it is a closure of that
     code where the static sets are bound
     (FlambdaClosuresSpec.project_function_slot_bound_statement); or
     the value of the symbol or constant in the value slot.

   The state may start from a static environment that binds something
   already and with continuations, where the code and the values that
   those hold have it ([state_sets_have_from_statement]): the static
   environment of the imports of a unit has no code and no closure, and
   what may follow a unit names no code id, so it makes no set.

   FlambdaClosuresSpec.sets_are is the test that every set which has
   a code anywhere is one given set; FlambdaKnownSlotsSpec.slots_as,
   that every set which has a value slot gives it one symbol or
   constant, whatever its code. This one goes by the code and the
   function slot, and asks one fact at a time.

   Nothing here is a rewrite. The proofs are in
   FlambdaSetsOfCodeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaKnownSymbolsSpec
  FlambdaKnownSlotsSpec FlambdaContextWhereSpec.
Import ListNotations.

(* ======================================================================== *)
(* The sets of closures that a term makes                                   *)
(* ======================================================================== *)

(* A set of closures as a term makes it: the code of its function
   slots, and what it puts in its value slots. *)
Definition set_made : Type :=
  (list (function_slot * code_id) * list (value_slot * simple))%type.

(* The set that a defining expression, or a static constant, makes. A
   set of one function with no function slot named is the set of that
   function at FlambdaValueSpec.unnamed_function_slot, as
   FlambdaValueSpec.closure_in_set reads its closure. *)
Definition named_sets_made (d : named) : list set_made :=
  match d with
  | Set_of_closures code value_slots =>
    [([(unnamed_function_slot, code)], value_slots)]
  | Closure_of_set _ function_decls value_slots =>
    [(function_decls, value_slots)]
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    []
  end.

Definition static_const_sets_made (const : static_const) : list set_made :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    [([(unnamed_function_slot, code)], value_slots)]
  | Static_const_closure_of_set _ _ function_decls value_slots =>
    [(function_decls, value_slots)]
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    []
  end.

(* All the sets that an expression makes, in handlers and in code
   too. *)
Fixpoint sets_made (e : expr) : list set_made :=
  match e with
  | Let _ _ d body => named_sets_made d ++ sets_made body
  | Let_static bindings body =>
    flat_map binding_sets_made bindings ++ sets_made body
  | Let_cont_non_recursive _ h body => handler_sets_made h ++ sets_made body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_sets_made h) handlers
    ++ sets_made body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_sets_made (h : continuation_handler) : list set_made :=
  match h with
  | Handler _ _ handler => sets_made handler
  end
with binding_sets_made (binding : static_binding) : list set_made :=
  match binding with
  | Bind_code _ c => code_sets_made c
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => static_const_sets_made const
  end
with code_sets_made (c : code) : list set_made :=
  match c with
  | Code _ _ _ _ _ body => sets_made body
  end.

(* ======================================================================== *)
(* What may be known of a set, and the test                                 *)
(* ======================================================================== *)

(* What may be known of a set of closures: the code of one of its
   function slots, or what one of its value slots holds. *)
Inductive set_fact : Type :=
  | Code_at (slot : function_slot) (code : code_id)
  | Slot_holds (slot : value_slot) (held : simple).

(* What a value slot is known to hold is a symbol or a constant. *)
Definition fact_closed (fact : set_fact) : bool :=
  match fact with
  | Code_at _ _ => true
  | Slot_holds _ held => closed_simple held
  end.

(* The function declarations have the code [c] at the function slot
   [f]. *)
Definition has_code_at (c : code_id) (f : function_slot)
    (function_decls : list (function_slot * code_id)) : bool :=
  match lookup f function_decls with
  | Some code => Pos.eqb code c
  | None => false
  end.

(* A set that a term makes has the fact. *)
Definition set_made_has (fact : set_fact) (made : set_made) : bool :=
  match fact with
  | Code_at slot code => has_code_at code slot (fst made)
  | Slot_holds slot held =>
    match lookup slot (snd made) with
    | Some a => same_simple a held
    | None => false
    end
  end.

(* Each of the sets that has [c] at [f] has the fact. *)
Definition sets_made_have (c : code_id) (f : function_slot)
    (fact : set_fact) (made : list set_made) : bool :=
  forallb
    (fun set => negb (has_code_at c f (fst set)) || set_made_has fact set)
    made.

(* ======================================================================== *)
(* What the test gives of a state                                           *)
(* ======================================================================== *)

(* A set that a closure is of, with the values of its value slots, has
   the fact: the value slot has the value of the symbol or constant
   (FlambdaKnownSlotsSpec.value_is). *)
Definition set_held_has (fact : set_fact)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) : bool :=
  match fact with
  | Code_at slot code => has_code_at code slot function_decls
  | Slot_holds slot held =>
    match lookup slot slot_values with
    | Some v => value_is held v
    | None => false
    end
  end.

(* The same of a set that has [c] at [f]; nothing is asked of another. *)
Definition set_held_as (c : code_id) (f : function_slot) (fact : set_fact)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) : bool :=
  negb (has_code_at c f function_decls)
  || set_held_has fact function_decls slot_values.

(* Every closure in the value is of a set that is so, through the
   values of value slots, the fields of blocks and partial
   applications. A symbol is so whatever it is bound to: what it is
   bound to is asked to be so where it is bound
   ([allocated_sets_have]). *)
Fixpoint value_sets_have (c : code_id) (f : function_slot) (fact : set_fact)
    (v : value) : bool :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ | Value_location _ =>
    true
  | Value_closure code slot_values =>
    set_held_as c f fact [(unnamed_function_slot, code)] slot_values
    && forallb (fun '(_, contents) => value_sets_have c f fact contents)
         slot_values
  | Value_closure_of_set _ function_decls slot_values =>
    set_held_as c f fact function_decls slot_values
    && forallb (fun '(_, contents) => value_sets_have c f fact contents)
         slot_values
  | Value_block _ fields | Value_immutable_array fields =>
    forallb (value_sets_have c f fact) fields
  | Value_partial_application callee args _ =>
    value_sets_have c f fact callee
    && forallb (value_sets_have c f fact) args
  end.

Definition allocated_sets_have (c : code_id) (f : function_slot)
    (fact : set_fact) (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure code slot_values =>
    set_held_as c f fact [(unnamed_function_slot, code)] slot_values
    && forallb (fun '(_, contents) => value_sets_have c f fact contents)
         slot_values
  | Static_closure_of_set _ _ function_decls slot_values =>
    set_held_as c f fact function_decls slot_values
    && forallb (fun '(_, contents) => value_sets_have c f fact contents)
         slot_values
  | Static_block _ fields => forallb (value_sets_have c f fact) fields
  | Static_immutable_array elements =>
    forallb (value_sets_have c f fact) elements
  | Static_boxed_number contents => value_sets_have c f fact contents
  | Static_string _ | Static_imported _ => true
  end.

(* The handlers that a continuation keeps, and those of the
   continuations that it keeps, make sets that pass the test. What is
   left of an over-application keeps no term: it is so when the two
   continuations that it keeps are. *)
Fixpoint cont_sets_have (c : code_id) (f : function_slot) (fact : set_fact)
    (k : cont_value) : Prop :=
  match k with
  | Outer _ => True
  | Closure h _ conts recursive =>
    sets_made_have c f fact (handler_sets_made h) = true
    ∧ forallb
        (fun '(_, h) => sets_made_have c f fact (handler_sets_made h))
        recursive
      = true
    ∧ fold_right (fun kc acc => cont_sets_have c f fact (snd kc) ∧ acc) True
        conts
  | Extra_args k' _ => cont_sets_have c f fact k'
  | Over_application _ _ _ return_to exn_to =>
    cont_sets_have c f fact return_to ∧ cont_sets_have c f fact exn_to
  end.

Definition conts_sets_have (c : code_id) (f : function_slot)
    (fact : set_fact) (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_sets_have c f fact (snd kc) ∧ acc) True
    conts.

(* The code that is bound makes sets that pass the test. *)
Definition code_sets_have (c : code_id) (f : function_slot) (fact : set_fact)
    (statics : static_env) : Prop :=
  ∀ (id : code_id) (code : code),
    lookup id (static_code statics) = Some code ->
    sets_made_have c f fact (code_sets_made code) = true.

(* Every set that the state holds and that has [c] at [f] has the fact:
   those that its terms make (its expression, the handlers that its
   continuations keep, the code that is bound), and those of the
   closures in its values (those of its variables, of the environments
   that its continuations keep and of the cells of its heap, and what
   each symbol is bound to; FlambdaSemanticsSpec.state_values_all). *)
Definition state_sets_have (c : code_id) (f : function_slot)
    (fact : set_fact) (s : state) : Prop :=
  sets_made_have c f fact (sets_made (state_expr s)) = true
  ∧ conts_sets_have c f fact (state_conts s)
  ∧ code_sets_have c f fact (state_statics s)
  ∧ state_values_all (fun _ v => value_sets_have c f fact v = true)
      (fun _ allocated => allocated_sets_have c f fact allocated = true) s.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The state that a unit starts in, run on its own, has it where the
   sets that the unit makes pass the test. Nothing else is asked of
   the unit; it need not be well-formed. *)
Definition state_sets_have_initial_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (fact : set_fact) (e : expr),
    sets_made_have c f fact (sets_made e) = true ->
    state_sets_have c f fact (State e [] [] empty_static_env empty_heap).

(* The same from a static environment that binds something already and
   with continuations: the code that the static environment has and
   the handlers that the continuations keep make sets that pass the
   test, and the values that they hold have it. The heap is empty. *)
Definition state_sets_have_from_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (fact : set_fact) (e : expr)
    (conts : cont_env) (statics : static_env),
    sets_made_have c f fact (sets_made e) = true ->
    conts_sets_have c f fact conts ->
    conts_values_all (fun v => value_sets_have c f fact v = true) conts ->
    code_sets_have c f fact statics ->
    Forall
      (fun binding => allocated_sets_have c f fact (snd binding) = true)
      (static_values statics) ->
    state_sets_have c f fact (State e [] conts statics empty_heap).

(* A step keeps it, where what a value slot is known to hold is a
   symbol or a constant. *)
Definition state_sets_have_step_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (fact : set_fact),
    fact_closed fact = true -> kept_by_step (state_sets_have c f fact).

(* In a state that has it, a closure that a variable is bound to, at
   the function slot [f] of a set that has [c] there, is of a set that
   has the fact. *)
Definition closure_set_has_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (fact : set_fact) (s : state)
    (x : variable) (v : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)),
    state_sets_have c f fact s ->
    lookup x (state_vars s) = Some v ->
    closure_in_set (state_statics s) v
    = Some (f, function_decls, slot_values) ->
    lookup f function_decls = Some c ->
    set_held_has fact function_decls slot_values = true.
