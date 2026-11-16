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



(* Examples of meets: what the transcribed functions answer on a few
   environments, and what satisfies the answers.

   This file is to be reviewed with FlambdaMeetSpec.v: each statement
   is a claim that its header makes, shown on the functions of
   FlambdaMeet.v by running them. The proofs are in
   FlambdaMeetExamplesProofs.v, which needs no review.

   - The step for [Get_tag] ("The step that is not sound"). The
     environment is the one that [Typing_env.add_get_tag_relation]
     makes of two variables ([example_get_tag_env_made_statement]).
     It is strictly satisfied where the scrutinee is an immediate, and
     the number 0 has the two types that are met
     ([example_get_tag_env_satisfied_statement]). The compiler's meet
     answers an environment that no valuation that keeps the value of
     the scrutinee satisfies ([compiler_meet_get_tag_statement]); so
     it has not [meet_sound], whatever the join, if this input is
     one that a meet may be given
     ([compiler_meet_not_sound_statement]). The meet that is to
     be proved answers an environment that the same valuation
     satisfies ([meet_to_prove_get_tag_statement]). It calls the join,
     the scrutinee being now an immediate or a block; the join given
     is the one below.
   - A join that has the contract of FlambdaMeetSpec.v: the one that
     forgets the environments joined and answers the environment of
     the meet ([forgetful_join_sound_statement]). The contract asks
     for no precision, and this shows that it asks for nothing that
     cannot be had. It is not the compiler's join. With it the meet
     that is to be proved has its contract with no premise left,
     whatever the fuel ([meet_with_forgetful_join_contract_statement],
     from the theorem of FlambdaMeetClosuresSpec.v), and so have the
     functions that add to a typing environment
     ([typing_env_functions_with_forgetful_join_statement]).
   - Adding an equation can forget ("The join that meet is given"):
     the environment answered has a valuation that the first has not
     ([add_equation_forgets_statement]). This is the compiler's
     behaviour, in both its files.
   - Poison ("What satisfies an environment"). Where a name holds
     poison and has a type that poison has not, the environment is
     satisfied with poison exempt, the value poison has an alias type
     to the name and the type of the values that are not null, and
     the meet of the two is bottom ([meet_through_poison_statement]).
     So the contract of a meet cannot be about
     [typing_env_satisfied].
   - The static environment ("What satisfies an environment"). A
     symbol is bound to a closure of a static set that has two
     function slots and names a symbol for the first only: the static
     environment is not valid. The symbol has two closures types, of
     which one says that the closure at the second slot is the number
     0 and the other that it is the number 1: the projection of the
     second slot from the symbol has no value, and a closures type
     asks nothing of a closure that the projection does not give. The
     meet of the two types is bottom
     ([meet_closures_invalid_statics_statement]). So the contracts
     that say that no value is lost suppose a static environment that
     is valid. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetFuelSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The step for [Get_tag]                                                   *)
(* ======================================================================== *)

Definition example_p : variable := 1%positive.
Definition example_tag : variable := 2%positive.

(* Two variables, [p] of kind value and [tag], a naked immediate. *)
Definition example_two_variables : Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_p Value Name_mode.normal
  in
  Typing_env.add_variable_definition tenv example_tag
    (Naked_number Kind_naked_immediate) Name_mode.normal.

(* The environment after [let tag = Get_tag p]: [tag] is [Get_tag] of
   [p], and [p] is a variant that has [tag] for its tag. *)
Definition example_get_tag_env : Typing_env.t :=
  let tenv :=
    Typing_env.replace_equation example_two_variables (Name_var example_tag)
      (get_tag_for_block (Name_var example_p))
  in
  Typing_env.replace_equation tenv (Name_var example_p)
    (Type_value
       (Ok
          (No_alias
             (Head_of_kind_value
                (Ok
                   (Variant None Or_unknown.Unknown (Some example_tag)
                      Or_unknown.Unknown No_extensions false))
                Not_null)))).

(* It is what [Typing_env.add_get_tag_relation] answers, whatever the
   join and the other parameters of the meet: with two variables that
   have just been defined there is nothing to meet. *)
Definition example_get_tag_env_made_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices),
    Flambda2_types.add_get_tag_relation join choices 1
      example_two_variables (Name_var example_tag) (Var example_p)
    = Some example_get_tag_env.

(* A run in which [p] is an immediate; unboxing passes 0 for the tag
   then. *)
Definition example_get_tag_valuation : env :=
  [(example_p, Value_const (Tagged_immediate 7));
   (example_tag, Value_const (Naked_immediate 0))].

(* The two types that are met, as where a switch on [tag] enters its
   arm for 0: the alias type of [tag], and the number 0. *)
Definition example_tag_type : flambda_type :=
  alias_type_of (Naked_number Kind_naked_immediate) (Var example_tag).

Definition example_zero_type : flambda_type := this_naked_immediate 0.

(* The environment is strictly satisfied in that run, and the value of
   [tag] has both types. *)
Definition example_get_tag_env_satisfied_statement : Prop :=
  typing_env_satisfied_strictly empty_static_env example_get_tag_valuation
    example_get_tag_env
  ∧ type_satisfied empty_static_env example_get_tag_valuation
      (Value_const (Naked_immediate 0)) example_tag_type
  ∧ type_satisfied empty_static_env example_get_tag_valuation
      (Value_const (Naked_immediate 0)) example_zero_type.

(* The compiler's meet answers a type and an environment, and no
   valuation that keeps the values of [p] and [tag] satisfies that
   environment: it has [p] a block. The join is not called. *)
Definition compiler_meet_get_tag_statement : Prop :=
  ∀ join : join_scoped_envs,
    ∃ (t : flambda_type) (tenv' : Typing_env.t),
      compiler_meet join 3 example_get_tag_env example_tag_type
        example_zero_type
      = Some (Or_bottom.Ok (t, tenv'))
      ∧ ∀ valuation' : env,
          extended_from example_get_tag_env example_get_tag_valuation
            valuation' ->
          ~ typing_env_satisfied_strictly empty_static_env valuation' tenv'.

(* So the compiler's meet has not [meet_sound], whatever the join, if
   this input is one that a meet may be given. *)
Definition compiler_meet_not_sound_statement : Prop :=
  ∀ join : join_scoped_envs,
    meet_inputs_well_formed example_get_tag_env example_tag_type
      example_zero_type ->
    ~ meet_sound (compiler_meet join 3).

(* ======================================================================== *)
(* A join that has its contract                                             *)
(* ======================================================================== *)

(* The join that forgets what the environments joined say: it answers
   the environment of the meet. *)
Definition forgetful_join : join_scoped_envs :=
  fun _meet_expanded_head _cut_after initial_env _initial_tenv _scoped_envs =>
    Some initial_env.

Definition forgetful_join_sound_statement : Prop :=
  join_scoped_envs_sound forgetful_join.

(* So the meet that is to be proved, with that join, has its contract:
   the premise of [transcription_sound] is one that a join has. *)
Definition meet_with_forgetful_join_contract_statement : Prop :=
  ∀ fuel : nat, meet_contract (meet_to_prove forgetful_join fuel).

(* The same of the functions that add to a typing environment, made of
   that meet. *)
Definition typing_env_functions_with_forgetful_join_statement : Prop :=
  ∀ fuel : nat,
    add_equation_sound
      (Flambda2_types.add_equation forgetful_join repaired_choices fuel)
    ∧ add_env_extension_sound
        (Flambda2_types.add_env_extension forgetful_join repaired_choices
           fuel)
    ∧ add_equations_on_params_sound_where_known
        (Flambda2_types.add_equations_on_params forgetful_join
           repaired_choices fuel).

(* With that join, the meet that is to be proved answers, on the input
   above, an environment that the run satisfies, and a type that the
   value of [tag] has. *)
Definition meet_to_prove_get_tag_statement : Prop :=
  ∃ (t : flambda_type) (tenv' : Typing_env.t),
    meet_to_prove forgetful_join 4 example_get_tag_env example_tag_type
      example_zero_type
    = Some (Or_bottom.Ok (t, tenv'))
    ∧ typing_env_satisfied_strictly empty_static_env
        example_get_tag_valuation tenv'
    ∧ type_satisfied empty_static_env example_get_tag_valuation
        (Value_const (Naked_immediate 0)) t.

(* ======================================================================== *)
(* Adding an equation can forget                                            *)
(* ======================================================================== *)

Definition example_x : variable := 1%positive.
Definition example_z : variable := 2%positive.

(* [x] is a value that may be null, and [z] says whether it is. *)
Definition example_is_null_env : Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_x Value Name_mode.normal
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_z
      (Naked_number Kind_naked_immediate) Name_mode.normal
  in
  Typing_env.replace_equation tenv (Name_var example_x)
    (Type_value
       (Ok
          (No_alias
             (Head_of_kind_value Unknown (Maybe_null (Some example_z)))))).

(* [x] is not null and [z] says that it is. *)
Definition example_is_null_valuation : env :=
  [(example_x, Value_const (Tagged_immediate 0));
   (example_z, Value_const (Naked_immediate 1))].

(* Adding that [x] is not null answers an environment that this
   valuation satisfies, and it does not satisfy the first: the
   environment answered no longer says what [z] is. The join is not
   called. *)
Definition add_equation_forgets_statement : Prop :=
  ∀ join : join_scoped_envs,
    ∃ tenv' : Typing_env.t,
      Flambda2_types.add_equation join compiler_choices 2
        example_is_null_env (Name_var example_x) any_non_null_value
      = Some tenv'
      ∧ typing_env_satisfied_strictly empty_static_env
          example_is_null_valuation tenv'
      ∧ ~ typing_env_satisfied_strictly empty_static_env
            example_is_null_valuation example_is_null_env.

(* ======================================================================== *)
(* Poison                                                                   *)
(* ======================================================================== *)

(* [x] is known to be null. *)
Definition example_null_env : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition Typing_env.create example_x Value
      Name_mode.normal
  in
  Typing_env.replace_equation tenv (Name_var example_x) null.

(* Where [x] holds poison the environment is satisfied with poison
   exempt; the value poison has the alias type of [x] and the type of
   the values that are not null; and the meet of the two is bottom. *)
Definition meet_through_poison_statement : Prop :=
  let valuation := [(example_x, Value_const (Poison Value))] in
  let alias := alias_type_of Value (Var example_x) in
  typing_env_satisfied empty_static_env valuation example_null_env
  ∧ type_satisfied empty_static_env valuation (Value_const (Poison Value))
      alias
  ∧ type_satisfied empty_static_env valuation (Value_const (Poison Value))
      any_non_null_value
  ∧ ∀ join : join_scoped_envs,
      compiler_meet join 2 example_null_env alias any_non_null_value
      = Some Or_bottom.Bottom.

(* ======================================================================== *)
(* A static environment that is not valid                                   *)
(* ======================================================================== *)

Definition example_slot_f : function_slot := 1%positive.
Definition example_slot_g : function_slot := 2%positive.
Definition example_set_symbol : symbol := 1%positive.

(* The functions of a set of closures that has two function slots. *)
Definition example_set_functions : list (function_slot * code_id) :=
  [(example_slot_f, 1%positive); (example_slot_g, 2%positive)].

(* The symbol is bound to the closure at the first slot of a static set
   with those functions, which names a symbol for the first slot and
   none for the second. *)
Definition example_invalid_statics : static_env :=
  Static_env []
    [(example_set_symbol,
      Static_closure_of_set example_slot_f
        [(example_slot_f, example_set_symbol)] example_set_functions [])].

(* A closure at the first slot of a set that has at least the two
   slots, in which the closure at the second slot is the number [n]. *)
Definition example_closures_type (n : Z) : flambda_type :=
  at_least_this_closure example_slot_f
    [(example_slot_f, Or_unknown.Unknown);
     (example_slot_g, Or_unknown.Unknown)]
    [(example_slot_g, this_tagged_immediate n)] [] Alloc_mode.Heap.

(* The static environment is not valid; the two types may be met in
   the environment that has nothing; the symbol has both; and the meet
   that is to be proved answers bottom, whatever the join. *)
Definition meet_closures_invalid_statics_statement : Prop :=
  ~ statics_valid example_invalid_statics
  ∧ meet_inputs_well_formed Typing_env.create (example_closures_type 0)
      (example_closures_type 1)
  ∧ typing_env_satisfied_strictly example_invalid_statics []
      Typing_env.create
  ∧ type_satisfied example_invalid_statics []
      (Value_symbol example_set_symbol) (example_closures_type 0)
  ∧ type_satisfied example_invalid_statics []
      (Value_symbol example_set_symbol) (example_closures_type 1)
  ∧ ∀ join : join_scoped_envs,
      meet_to_prove join 3 Typing_env.create (example_closures_type 0)
        (example_closures_type 1)
      = Some Or_bottom.Bottom.
