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


(* Examples of typing environments, with the environments of the machine
   that satisfy them and some that do not.

   This file is to be reviewed with FlambdaTypingEnvSpec.v: it is there
   to show that what that file states does not hold vacuously, and what
   it says in the cases that its header singles out (a name that has no
   value at run time, a variable that stands for nothing in a run, a
   symbol, code, poison). An environment that is satisfied is shown to
   be so strictly where it is; one that is not is shown not to be even
   with the names that hold poison exempt. The proofs are in
   FlambdaTypingEnvExamplesProofs.v, which needs no review.

   The first example is the environment that Simplify has built when it
   reaches the last line of

     let x = p in
     let c = <closure of code f, whose value slot w holds x> in
     let y = <the value slot w of c> in
     apply_cont k y

   where [p] is a parameter known to be the tagged immediate 7 or 8. It
   is built with the functions of FlambdaTypingEnv.v, called as the
   compiler calls them:
   - [p] is defined and given its type;
   - [x] is defined with the alias type [= p]: the aliases make [p] the
     canonical element of the two, [p] being bound earlier, and the type
     of [x], the name demoted, becomes [= p];
   - [c] is defined with the type of the closures of exactly that set,
     in which [x] has been replaced by its canonical element [p];
   - [y] is defined with the type of the value slot, [= p], as [x] was.
   Then [Typing_env.type_simple_in_term_exn] answers [p] for [y], so the
   last line is [apply_cont k p]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What Simplify builds for a small term                                    *)
(* ======================================================================== *)

Definition example_p : variable := 1%positive.
Definition example_x : variable := 2%positive.
Definition example_c : variable := 3%positive.
Definition example_y : variable := 4%positive.
Definition example_f : code_id := 1%positive.
Definition example_w : value_slot := 1%positive.

(* What [Meet_env.add_equation] comes to when it gives the alias type
   [= s] to a variable that has just been defined, whose type is still
   unknown: the aliases record that the two are equal
   ([Typing_env.add_alias]), and the name demoted gets the alias type of
   the canonical element ([Meet_env.record_demotion]; there is no type
   to meet). The two simples given are to be canonical. *)
Definition add_alias_of_new_variable (tenv : Typing_env.t) (x : variable)
    (kind : flambda_kind) (s : simple) : option Typing_env.t :=
  match Typing_env.add_alias tenv (Var x) s with
  | Some (Ok result) =>
    Some
      (Typing_env.replace_equation (Typing_env.add_alias_result_t result)
         (Typing_env.demoted_name result)
         (alias_type_of kind (Typing_env.canonical_element result)))
  | Some Unknown | Some Bottom | None => None
  end.

(* The type that Simplify gives [c]
   ([Simplify_set_of_closures]): the closure at the one function slot of
   a set with exactly that function, of code [f], and the value slot
   [w], which holds [p]. *)
Definition example_closure_type : flambda_type :=
  exactly_this_closure unnamed_function_slot
    [(unnamed_function_slot, Or_unknown.Known (Function_type example_f))]
    [(unnamed_function_slot, alias_type_of Value (Var example_c))]
    [(example_w, alias_type_of Value (Var example_p))]
    Alloc_mode.Heap_or_local.

Definition example_env : option Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_p Value Name_mode.normal
  in
  let tenv :=
    Typing_env.replace_equation tenv (Name_var example_p)
      (these_tagged_immediates [7; 8])
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_x Value Name_mode.normal
  in
  let* tenv :=
    add_alias_of_new_variable tenv example_x Value (Var example_p)
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_c Value Name_mode.normal
  in
  let tenv :=
    Typing_env.replace_equation tenv (Name_var example_c)
      example_closure_type
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_y Value Name_mode.normal
  in
  add_alias_of_new_variable tenv example_y Value (Var example_p).

(* The variables of the machine at the last line, where [p] holds [vp]
   and [y] holds [vy]. *)
Definition example_vars (vp vy : value) : env :=
  [(example_y, vy);
   (example_c, Value_closure example_f [(example_w, vp)]);
   (example_x, vp);
   (example_p, vp)].

Definition tagged (i : Z) : value := Value_const (Tagged_immediate i).

(* The environment is built. *)
Definition example_env_built_statement : Prop :=
  ∃ tenv : Typing_env.t, example_env = Some tenv.

(* It is satisfied, strictly, by the machine where [p] holds 7 and the
   term has run to its last line, with nothing in the static
   environment: the code [f] of the closure is not bound there, and
   nothing is asked of it. *)
Definition example_env_satisfied_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    env_satisfies_strictly tenv empty_static_env
      (example_vars (tagged 7) (tagged 7)).

(* It is not where [p] holds 9, which is not of its type... *)
Definition example_env_wrong_type_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    ~ env_satisfies tenv empty_static_env
        (example_vars (tagged 9) (tagged 9)).

(* ... nor where [y] and [p] hold different values, though each is 7
   or 8: the type of [y] is the alias of [p], and the aliases say that
   the two are equal... *)
Definition example_env_wrong_alias_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    ~ env_satisfies tenv empty_static_env
        (example_vars (tagged 7) (tagged 8)).

(* ... nor where a variable in scope has no value. *)
Definition example_env_unbound_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    ~ env_satisfies tenv empty_static_env [(example_p, tagged 7)].

(* The canonical element of [y] is [p], and its type the alias to it:
   Simplify writes [p] for [y]. *)
Definition example_env_canonical_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal)
      (Var example_y)
    = Some (Some (alias_type_of Value (Var example_p), Var example_p)).

(* Poison. Where [p] holds poison, as a parameter that stands for a
   field of an inactive case of a variant does, nothing is asked of its
   type if the names that hold poison are exempt: the environment is
   satisfied where the other names hold what they must, the same
   poison... *)
Definition poison (k : flambda_kind) : value := Value_const (Poison k).

Definition example_env_poison_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    env_satisfies tenv empty_static_env
      (example_vars (poison Value) (poison Value)).

(* ... but not strictly... *)
Definition example_env_poison_not_strictly_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    ~ env_satisfies_strictly tenv empty_static_env
        (example_vars (poison Value) (poison Value)).

(* ... and there the statements about types are not to be had: poison
   has the alias type of [p], and has not the type that the environment
   finds for [p]. So an alias to an exempt name may not be replaced by
   the type of the name. *)
Definition example_env_poison_alias_type_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    type_satisfied empty_static_env
      (example_vars (poison Value) (poison Value)) (poison Value)
      (alias_type_of Value (Var example_p))
    ∧ Typing_env.find tenv (Name_var example_p) None
      = Some (these_tagged_immediates [7; 8])
    ∧ ~ type_satisfied empty_static_env
          (example_vars (poison Value) (poison Value)) (poison Value)
          (these_tagged_immediates [7; 8]).

(* A relation on a name that holds poison. Where a field of an inactive
   case of a variant is unboxed in turn, unboxing passes poison for the
   field and 0 for the parameter that stands for [Is_int] of it, whose
   type is that relation. Here [p] is the field and [i] the parameter.
   Where [p] holds an immediate and [i] holds 1 the environment is
   satisfied strictly... *)
Definition example_i : variable := 8%positive.

Definition example_relation_env : Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_p Value Name_mode.normal
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_i
      (Naked_number Kind_naked_immediate) Name_mode.normal
  in
  Typing_env.replace_equation tenv (Name_var example_i)
    (is_int_for_scrutinee (Name_var example_p)).

Definition example_relation_env_immediate_statement : Prop :=
  env_satisfies_strictly example_relation_env empty_static_env
    [(example_i, Value_const (Naked_immediate 1)); (example_p, tagged 5)].

(* ... and where [p] holds poison and [i] holds 0 it is satisfied with
   poison exempt: no relation is true of a scrutinee that holds poison,
   and [i], whose type is a relation on [p], is exempt with [p]. *)
Definition example_relation_env_poison_statement : Prop :=
  env_satisfies example_relation_env empty_static_env
    [(example_i, Value_const (Naked_immediate 0)); (example_p, poison Value)].

(* Back to the first environment: it is not satisfied where [y], an
   alias of [p], holds something else than [p]: the aliases are exact,
   poison or not... *)
Definition example_env_poison_alias_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_env = Some tenv ->
    ~ env_satisfies tenv empty_static_env
        (example_vars (poison Value) (tagged 7)).

(* ... nor where [p] holds the poison of another kind than its own. *)
Definition example_env_poison_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (v : value),
    example_env = Some tenv ->
    v = poison (Naked_number Kind_naked_immediate) ->
    ~ env_satisfies tenv empty_static_env (example_vars v v).

(* ======================================================================== *)
(* A name that has no value at run time                                     *)
(* ======================================================================== *)

(* The environment in which the body of the code [f] is simplified: the
   one above, where the closure is made, with the variables bound so far
   out of scope ([Typing_env.closure_env]) and the closure being called
   defined, with the type of [c]. *)
Definition example_my_closure : variable := 5%positive.

Definition example_body_env : option Typing_env.t :=
  let* tenv := example_env in
  let tenv := Typing_env.closure_env tenv in
  let tenv :=
    Typing_env.add_variable_definition tenv example_my_closure Value
      Name_mode.normal
  in
  add_alias_of_new_variable tenv example_my_closure Value (Var example_c).

(* In the body the machine has the closure called and no other
   variable: [p], [x], [c] and [y] have no value there. *)
Definition example_body_vars (v : value) : env :=
  [(example_my_closure, Value_closure example_f [(example_w, v)])].

Definition example_body_env_built_statement : Prop :=
  ∃ tenv : Typing_env.t, example_body_env = Some tenv.

(* [p] is bound in the environment and may not occur in a term. *)
Definition example_body_env_scope_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_body_env = Some tenv ->
    Typing_env.mem None tenv (Name_var example_p) = true
    ∧ ~ in_scope_in_terms tenv example_p
    ∧ in_scope_in_terms tenv example_my_closure.

(* The environment is satisfied, strictly, where the value slot of the
   closure called holds 7: the valuation gives [p] that value... *)
Definition example_body_env_satisfied_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_body_env = Some tenv ->
    env_satisfies_strictly tenv empty_static_env
      (example_body_vars (tagged 7)).

(* ... and is not where it holds 9: [p] would have to be 9, as the slot
   is, and its type says 7 or 8. So what is known of a variable that is
   out of scope constrains the values that are in scope. *)
Definition example_body_env_wrong_slot_statement : Prop :=
  ∀ tenv : Typing_env.t,
    example_body_env = Some tenv ->
    ~ env_satisfies tenv empty_static_env (example_body_vars (tagged 9)).

(* ======================================================================== *)
(* A variable that stands for nothing in a run                              *)
(* ======================================================================== *)

(* The environment of the handler of a continuation with one parameter
   [q], which one use calls with the immediate 0 and another with a
   block of tag 0 whose one field is a variable bound at that use,
   known there to be the tagged immediate 4 or 5. The join brings that
   variable in, [e], in types only and with its type, and gives [q] the
   type "the immediate 0, or a block of tag 0 whose field is [e]". (The
   differential tests of the types run the compiler's two joins on
   this: both answer so; the n-way join also gives the variant a
   variable for [Is_int] of [q].) *)
Definition example_q : variable := 6%positive.
Definition example_e : variable := 7%positive.

Definition example_join_type : flambda_type :=
  create_variant false
    (Or_unknown.Known (these_naked_immediates [0]))
    (Or_unknown.Known
       (row_like_for_blocks_create_exactly 0 1 (Scannable Value_only)
          [alias_type_of Value (Var example_e)] Alloc_mode.Heap))
    No_extensions.

Definition example_join_env : Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_q Value Name_mode.normal
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_e Value
      Name_mode.in_types
  in
  let tenv :=
    Typing_env.replace_equation tenv (Name_var example_e)
      (these_tagged_immediates [4; 5])
  in
  Typing_env.replace_equation tenv (Name_var example_q) example_join_type.

(* In a run that comes through the first use, [q] holds 0 and [e]
   stands for nothing: the environment is satisfied, strictly, by the
   valuation that gives [e] no value. (Here a value of the type of [e]
   exists and would do as well; in general none need exist in such a
   run.) *)
Definition example_join_env_immediate_statement : Prop :=
  env_satisfies_strictly example_join_env empty_static_env
    [(example_q, tagged 0)].

(* In a run that comes through the second, [q] holds a block whose
   field is 4, and the valuation gives [e] that value... *)
Definition example_join_env_block_statement : Prop :=
  env_satisfies_strictly example_join_env empty_static_env
    [(example_q, Value_block 0 [tagged 4])].

(* ... and the environment is not satisfied where the field is 6: [e]
   would have to be 6, as the field is, and its type says 4 or 5. *)
Definition example_join_env_wrong_block_statement : Prop :=
  ~ env_satisfies example_join_env empty_static_env
      [(example_q, Value_block 0 [tagged 6])].

(* An environment that is not bottom, in which a variable that exists
   only in types has the type bottom. [Meet_env] makes such
   environments when it is told not to raise on bottom
   ([add_concrete_equation_on_canonical_name], which then gives the name
   the type bottom with [Typing_env.replace_equation]); the binary join
   works in one. It is satisfied, strictly, by the valuations that give
   the variable no value, and by no other. So that the type of a name
   is bottom says that a program point is not reached only of a name
   that has a value there. *)
Definition example_absent_env : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create example_e Value
       Name_mode.in_types)
    (Name_var example_e) (bottom Value).

Definition example_absent_env_satisfied_statement : Prop :=
  ∀ (statics : static_env) (valuation : env),
    typing_env_satisfied_strictly statics valuation example_absent_env
    <-> lookup example_e valuation = None.

(* ======================================================================== *)
(* A symbol                                                                 *)
(* ======================================================================== *)

Definition example_s : symbol := 1%positive.

(* A symbol that is defined and has no equation: its type is that of the
   values that are not null, which a symbol is whatever the static
   environment binds it to, or does not. *)
Definition example_defined_symbol_statement : Prop :=
  ∀ (statics : static_env) (vars : env),
    env_satisfies_strictly
      (Typing_env.add_symbol_definition Typing_env.create example_s) statics
      vars.

(* The same symbol with the type of the blocks of tag 0 whose one field
   is the tagged immediate 1. *)
Definition example_symbol_env : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_symbol_definition Typing_env.create example_s)
    (Name_symbol example_s)
    (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
       [these_tagged_immediates [1]]).

(* It is satisfied where the static environment binds the symbol to such
   a block, and not where it does not bind it. *)
Definition example_symbol_env_satisfied_statement : Prop :=
  env_satisfies_strictly example_symbol_env
    (Static_env [] [(example_s, Static_block 0 [tagged 1])]) [].

Definition example_symbol_env_unbound_statement : Prop :=
  ∀ vars : env, ~ env_satisfies example_symbol_env empty_static_env vars.
