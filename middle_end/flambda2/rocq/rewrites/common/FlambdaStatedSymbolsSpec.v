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

(* What is known of the symbols that the [Let_static]s around a point
   of a term have bound.

   This file is to be reviewed: the definitions and the statements at
   the end. The proofs are in FlambdaStatedSymbolsProofs.v, which
   needs no review.

   FlambdaStatedScopeSpec.v knows of a symbol only that it is no null
   and no poison. A handler or code may state more of a parameter: a
   shape of block. Simplify lifts a block of constants to a symbol
   and gives the symbol where the block was given, so the term after
   Simplify gives a symbol where the binder states a shape, and the
   tests of FlambdaStatedScopeSpec.v refuse. This file lets a rewrite
   take the test of the machine (decision D25, at parameters) away
   where it can establish that the symbol is bound to a block of the
   shape stated. As there, that is a sufficient reason and is not said
   to be the weakest; the rest of the equivalence is the rewrite's own
   proof.

   What is known of a symbol comes from the term, going down it
   ([symbols_stated], the most recently bound first): a [Let_static]
   adds an entry for each symbol that it binds ([statics_stated]).
   A symbol bound to a block of values, each of whose fields is given
   as a symbol or as a tagged immediate, is known as a block of that
   tag with what is known of each field: of a field that is a symbol,
   what the list knows of that symbol so far, so that a symbol bound
   earlier in the same group is known and one bound later is not; of
   a field that is a tagged immediate, what
   FlambdaStatedScopeSpec.stated_of_simple says of the constant. A
   block with any other field (a variable, another constant) is known
   as no null only: the machine does not test the kind of a field of
   a static block (FlambdaSemanticsSpec.eval_static_const), and the
   test of a shape asks that each field be of kind [Value]
   (FlambdaSubkindSpec.fields_of_subkinds). Every other static
   constant is known as no null. Of a symbol with no entry (one that
   the term does not bind above the point) the same is known.

   The test ([are_stated_knowing]) is
   FlambdaStatedScopeSpec.are_stated, but that a symbol is known as
   what the list says of it. With the empty list it is that test.

   The invariant ([symbols_hold]): each symbol of the list is, in the
   static environment of the machine, a value of what the list says
   (FlambdaSubkindSpec.value_at). The statements say that the machine
   keeps it: the step of a [Let_static] binds the symbols to what
   [statics_stated] says ([statics_stated_hold_statement]), and the
   static environment only grows afterwards, which keeps what a value
   is of ([symbols_hold_extend_statement]). The machine has no step
   on a [Let_static] that binds a symbol bound already
   (FlambdaSemanticsSpec.eval_static_bindings), so an entry is never
   about another binding than the one that the machine made. And
   where the invariant holds with that of the variables
   (FlambdaStatedScopeSpec.stated_hold), the test is the test of the
   machine ([are_stated_knowing_sound_statement]).

   Not covered: as in FlambdaStatedScopeSpec.v, the statements are of
   single steps; that a rewrite which threads the list down a term
   meets, at each node, a static environment in which it holds is
   part of the proof of that rewrite. The body of code is entered
   under a static environment that extends the one in which the code
   was bound, so what is known of the symbols where the code is bound
   is known in its body. The list is not used to say more of what a
   defining expression makes (FlambdaStatedScopeSpec.stated_of_named):
   a variable bound to a symbol by a [Let] is still known as no null
   only, and a load keeps the rules of that file: a load from a
   symbol whose access kind says any value gives nothing, not even
   that the field is no null, and one whose access kind says an
   immediate gives a tagged immediate. This list adds nothing to
   those. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSubkindSpec
  FlambdaSubkindOrderSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaStatedScopeSpec.
Import ListNotations.

Definition symbols_stated : Type := list (symbol * param_subkind).

Definition stated_of_symbol (Y : symbols_stated) (sym : symbol)
    : param_subkind :=
  match lookup sym Y with
  | Some ps => ps
  | None => not_null
  end.

(* What is known of a field of a static block, where the field is
   given as a symbol or as a tagged immediate: its value is then of
   kind [Value]. Of any other field nothing is said, not even its
   kind. *)
Definition static_field_stated (Y : symbols_stated) (s : simple)
    : option param_subkind :=
  match s with
  | Symbol sym => Some (stated_of_symbol Y sym)
  | Const (Tagged_immediate _) => Some (stated_of_simple [] s)
  | Var _ | Const _ => None
  end.

Fixpoint static_fields_stated (Y : symbols_stated) (fields : list simple)
    : option field_subkinds :=
  match fields with
  | [] => Some Fields_nil
  | s :: fields =>
    let* ps := static_field_stated Y s in
    let* rest := static_fields_stated Y fields in
    Some (Fields_cons (fst ps) (snd ps) rest)
  end.

(* What a symbol bound to a static constant is known to be. *)
Definition stated_of_static_const (Y : symbols_stated) (const : static_const)
    : param_subkind :=
  match const with
  | Static_const_block tag fields =>
    match static_fields_stated Y fields with
    | Some fields =>
      (false, Subkind_variant [] (Shape_determined tag fields Shapes_nil))
    | None => not_null
    end
  | Static_const_set_of_closures _ _ | Static_const_closure_of_set _ _ _ _
  | Static_const_boxed_number _ | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    not_null
  end.

(* What a group of bindings adds to what is known of the symbols, the
   last bound first. *)
Fixpoint statics_stated (bindings : list static_binding)
    (Y : symbols_stated) : symbols_stated :=
  match bindings with
  | [] => Y
  | Bind_symbol sym const :: bindings =>
    statics_stated bindings ((sym, stated_of_static_const Y const) :: Y)
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    statics_stated bindings Y
  end.

(* Whether a [Simple] is known to be of what a binder states, a symbol
   by what is known of the symbols. *)
Definition is_stated_knowing (Y : symbols_stated) (S : stated_in_scope)
    (ps : param_subkind) (s : simple) : bool :=
  match s with
  | Symbol sym => param_says_at_least (stated_of_symbol Y sym) ps
  | Var _ | Const _ => is_stated S ps s
  end.

(* FlambdaStatedScopeSpec.are_stated with that test. *)
Fixpoint are_stated_knowing (Y : symbols_stated) (S : stated_in_scope)
    (args : list simple) (sks : list param_subkind) {struct sks} : bool :=
  match sks, args with
  | ps :: sks, s :: args =>
    is_stated_knowing Y S ps s && are_stated_knowing Y S args sks
  | _, _ => true
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What is known of the symbols holds in a static environment: each
   symbol of the list is a value of what the list says of it. Of two
   entries for one symbol the one bound last counts. *)
Definition symbols_hold (statics : static_env) (Y : symbols_stated) : Prop :=
  ∀ (sym : symbol) (ps : param_subkind),
    lookup sym Y = Some ps ->
    value_at statics ps (Value_symbol sym) = true.

Definition symbols_hold_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (Y : symbols_stated),
    statics_extend statics statics' ->
    symbols_hold statics Y ->
    symbols_hold statics' Y.

(* The step of a [Let_static] keeps what is known, with the entries
   that the rewrites add. *)
Definition statics_stated_hold_statement : Prop :=
  ∀ (Y : symbols_stated) (bindings : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) (s : state),
    symbols_hold statics Y ->
    step (State (Let_static bindings body) vars conts statics hp rd) = Step s ->
    symbols_hold (state_statics s) (statics_stated bindings Y).

Definition is_stated_knowing_sound_statement : Prop :=
  ∀ (statics : static_env) (Y : symbols_stated) (S : stated_in_scope)
    (vars : env) (ps : param_subkind) (s : simple) (v : value),
    symbols_hold statics Y ->
    stated_hold statics S vars ->
    is_stated_knowing Y S ps s = true ->
    eval_simple vars s = Some v ->
    value_at statics ps v = true.

Definition are_stated_knowing_sound_statement : Prop :=
  ∀ (statics : static_env) (Y : symbols_stated) (S : stated_in_scope)
    (vars : env) (args : list simple) (sks : list param_subkind)
    (vs : list value),
    symbols_hold statics Y ->
    stated_hold statics S vars ->
    are_stated_knowing Y S args sks = true ->
    eval_simples vars args = Some vs ->
    values_of_subkinds statics vs sks = true.
