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


(* The functions of the environment of the meet commute with a renaming
   of variables. Lemmas only: nothing here is a statement to review.

   For each function of the module [Meet_env] of FlambdaMeetEnv.v, and
   for the functions of the modules [Meet] and [Flambda2_types] at the
   end of FlambdaMeet.v (the meet as it is exposed, and the functions
   that add an equation to a typing environment), a lemma named after
   the function says that on the renamed environment and renamed
   arguments it gives the renamed answer. The action on what the meet
   takes and answers is that of FlambdaMeetRenameSpec.v. A function
   that is given the meet of expanded types is given two, one with the
   arguments and one with the renamed arguments, which the renaming
   relates ([meet_expanded_head_related]): the second answers of
   renamed types the renamed answer of the first. Where the two are
   one, that is that the meet commutes with the renaming. The same of
   the functions that are given the join. The laws used are those of
   the typing environment (FlambdaEnvRenameProofs.v), of the aliases,
   of types and of expanded types. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaTypingEnv FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaProversRenameSpec FlambdaProversRenameProofs FlambdaMeetEnv
  FlambdaMeet FlambdaMeetSpec FlambdaMeetRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* Types and expanded types                                                 *)
(* ======================================================================== *)

Lemma to_type_renamed (r : renaming) (expanded : Expanded_type.t) :
  Expanded_type.to_type (rename_expanded r expanded)
  = rename_type r (Expanded_type.to_type expanded).
Proof.
  destruct expanded as [k [ | [head | head | k' head | head | head] | ]];
    cbn [Expanded_type.to_type rename_expanded Expanded_type.descr
         Expanded_type.kind or_unknown_or_bottom_map rename_expanded_head];
    try reflexivity.
  - symmetry. apply (proj1 (rename_type_closed r)).
  - symmetry. apply (proj1 (proj2 (rename_type_closed r))).
Qed.

Lemma kind_of_expanded_renamed (r : renaming) (expanded : Expanded_type.t) :
  Expanded_type.kind (rename_expanded r expanded)
  = Expanded_type.kind expanded.
Proof. reflexivity. Qed.

Lemma expand_head0_renamed (r : renaming) (tenv : Typing_env.t)
    (ty : flambda_type) (known : option simple) :
  expand_head0 (rename_typing_env r tenv) (rename_type r ty)
    (option_map (rename_simple r) known)
  = option_map (rename_expanded r) (expand_head0 tenv ty known).
Proof.
  unfold expand_head0.
  rewrite (proj2 (rename_type_alias r)), kind_rename_type.
  destruct (get_alias_opt ty) as [alias | ]; cbn [option_map];
    [ | apply of_non_alias_type_renamed ].
  destruct known as [simple | ]; cbn [option_map];
    [ apply expand_head_of_alias_type_renamed
    | apply of_non_alias_type_unknown ].
Qed.

(* [FlambdaMeetGrammar.kind_of_simple]. *)
Lemma kind_of_simple_renamed (r : renaming) (tenv : Typing_env.t)
    (simple : simple) :
  kind_of_simple (rename_typing_env r tenv) (rename_simple r simple)
  = kind_of_simple tenv simple.
Proof.
  destruct simple as [x | sym | c]; cbn [kind_of_simple rename_simple];
    try reflexivity.
  change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
  rewrite find_equivariant.
  destruct (Typing_env.find tenv (Name_var x) None) as [ty | ];
    cbn [option_map]; [ rewrite kind_rename_type | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* What a meet returns                                                      *)
(* ======================================================================== *)

Lemma extract_value_renamed {A : Type} (f : A -> A)
    (res : meet_return_value A) (left right : A) :
  extract_value (rename_return_value f res) (f left) (f right)
  = f (extract_value res left right).
Proof. destruct res; reflexivity. Qed.

Lemma map_result_renamed {A B : Type} (r : renaming) (f : A -> A)
    (g : B -> B) (h : A -> B) (result : Meet_result.t A) :
  (∀ a : A, h (f a) = g (h a)) ->
  map_result h (rename_meet_result r f result)
  = rename_meet_result r g (map_result h result).
Proof.
  intros Hh. destruct result as [res | [ | | | a] env];
    cbn [map_result rename_meet_result rename_return_value];
    try reflexivity.
  rewrite Hh. reflexivity.
Qed.

(* ======================================================================== *)
(* The environment of the meet                                              *)
(* ======================================================================== *)

Lemma create_renamed (r : renaming) (tenv : Typing_env.t) :
  Meet_env.create (rename_typing_env r tenv)
  = rename_meet_env r (Meet_env.create tenv).
Proof. reflexivity. Qed.

Lemma with_typing_env_renamed (r : renaming) (env : Meet_env.t)
    (tenv : Typing_env.t) :
  Meet_env.with_typing_env (rename_meet_env r env) (rename_typing_env r tenv)
  = rename_meet_env r (Meet_env.with_typing_env env tenv).
Proof. reflexivity. Qed.

Lemma map_typing_env_renamed (r : renaming) (env : Meet_env.t)
    (f f' : Typing_env.t -> Typing_env.t) :
  (∀ tenv : Typing_env.t,
    f' (rename_typing_env r tenv) = rename_typing_env r (f tenv)) ->
  Meet_env.map_typing_env (rename_meet_env r env) f'
  = rename_meet_env r (Meet_env.map_typing_env env f).
Proof.
  intros Hf. unfold Meet_env.map_typing_env.
  rewrite <- with_typing_env_renamed, <- Hf. reflexivity.
Qed.

Lemma with_names_to_avoid_renamed (r : renaming) (env : Meet_env.t)
    (names : list variable) :
  Meet_env.with_names_to_avoid (rename_meet_env r env)
    (map (rename_variable r) names)
  = rename_meet_env r (Meet_env.with_names_to_avoid env names).
Proof. reflexivity. Qed.

Lemma current_typing_env_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.current_typing_env (rename_meet_env r env)
  = rename_typing_env r (Meet_env.current_typing_env env).
Proof. reflexivity. Qed.

Lemma make_bottom_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.make_bottom (rename_meet_env r env)
  = rename_meet_env r (Meet_env.make_bottom env).
Proof.
  apply map_typing_env_renamed. intros tenv. apply make_bottom_equivariant.
Qed.

Lemma final_typing_env_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.final_typing_env (rename_meet_env r env)
  = rename_typing_env r (Meet_env.final_typing_env env).
Proof. reflexivity. Qed.

Lemma final_typing_env_strict_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.final_typing_env_strict (rename_meet_env r env)
  = or_bottom_map (rename_typing_env r)
      (Meet_env.final_typing_env_strict env).
Proof. reflexivity. Qed.

Lemma enter_scope_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.enter_scope (rename_meet_env r env)
  = (fst (fst (Meet_env.enter_scope env)),
     rename_typing_env r (snd (fst (Meet_env.enter_scope env))),
     rename_meet_env r (snd (Meet_env.enter_scope env))).
Proof.
  unfold Meet_env.enter_scope. cbv zeta. cbn [fst snd].
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite current_scope_equivariant, increment_scope_equivariant.
  reflexivity.
Qed.

Lemma add_definition_renamed (r : renaming) (env : Meet_env.t) (name : name)
    (name_mode : Name_mode.t) (kind : flambda_kind) :
  Meet_env.add_definition (rename_meet_env r env) (rename_name r name)
    name_mode kind
  = option_map (rename_meet_env r)
      (Meet_env.add_definition env name name_mode kind).
Proof.
  unfold Meet_env.add_definition.
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite add_definition_equivariant.
  destruct
    (Typing_env.add_definition (Meet_env.typing_env env) name name_mode kind);
    reflexivity.
Qed.

Lemma add_variable_definition_renamed (r : renaming) (env : Meet_env.t)
    (var : variable) (kind : flambda_kind) (name_mode : Name_mode.t) :
  Meet_env.add_variable_definition (rename_meet_env r env)
    (rename_variable r var) kind name_mode
  = rename_meet_env r
      (Meet_env.add_variable_definition env var kind name_mode).
Proof.
  apply map_typing_env_renamed. intros tenv.
  apply add_variable_definition_equivariant.
Qed.

(* ======================================================================== *)
(* An equation added to a name                                              *)
(* ======================================================================== *)

Lemma replace_concrete_equation_renamed (r : renaming) (tenv : Typing_env.t)
    (name : name) (ty : flambda_type) :
  Meet_env.replace_concrete_equation (rename_typing_env r tenv)
    (rename_name r name) (rename_type r ty)
  = option_map (rename_typing_env r)
      (Meet_env.replace_concrete_equation tenv name ty).
Proof.
  unfold Meet_env.replace_concrete_equation.
  rewrite must_be_singleton_rename_type.
  destruct (must_be_singleton ty) as [[const | ] | ]; [ | | reflexivity ].
  - rewrite aliases_equivariant, <- rename_simple_of_name, canonical_renamed.
    destruct (Aliases.canonical (Typing_env.aliases tenv)
                (simple_of_name name)) as [ | canonical_element];
      cbn [rename_is_canonical_or_alias]; [ | reflexivity ].
    change (Const const) with (rename_simple r (Const const)) at 1.
    rewrite add_alias_equivariant.
    destruct (Typing_env.add_alias tenv (simple_of_name name) (Const const))
      as [[ | result | ] | ];
      cbn [option_map or_unknown_or_bottom_map]; try reflexivity.
    cbn [rename_add_alias_result Typing_env.demoted_name
         Typing_env.canonical_element Typing_env.add_alias_result_t].
    rewrite name_eqb_renamed.
    change (Const const) with (rename_simple r (Const const)) at 1.
    rewrite simple_equal_renamed.
    destruct
      (name_eqb (Typing_env.demoted_name result) name
       && simple_equal (Typing_env.canonical_element result) (Const const));
      [ | reflexivity ].
    cbn [option_map]. f_equal.
    rewrite <- (proj1 (rename_type_alias r)).
    apply replace_equation_equivariant.
  - rewrite (proj2 (rename_type_alias r)).
    destruct (get_alias_opt ty) as [alias | ]; cbn [option_map];
      [ reflexivity | ].
    f_equal. apply replace_equation_equivariant.
Qed.

(* The equations of a list, renamed on both sides. *)
Definition rename_equations (r : renaming)
    (equations : list (name * flambda_type)) : list (name * flambda_type) :=
  map (fun equation => (rename_name r (fst equation),
                        rename_type r (snd equation)))
    equations.

(* Parameters, renamed; their kinds are kept. *)
Definition rename_params (r : renaming)
    (params : list (variable * flambda_kind))
    : list (variable * flambda_kind) :=
  map (fun param => (rename_variable r (fst param), snd param)) params.

Section Meet_given.

(* One renaming and two meets of expanded types: the second, which the
   functions are given with the renamed arguments, answers of renamed
   types the renamed answer of the first. Where the two are one meet,
   this is that the meet commutes with the renaming. *)
Variable r : renaming.
Variables m m' : meet_expanded_head.
Hypothesis m_related : meet_expanded_head_related r m m'.

Lemma add_concrete_equation_on_canonical_name_renamed
    (raise_on_bottom : bool) (env : Meet_env.t) (name : name)
    (ty : Expanded_type.t) :
  Meet_env.add_concrete_equation_on_canonical_name raise_on_bottom
    (rename_meet_env r env) (rename_name r name) (rename_expanded r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_concrete_equation_on_canonical_name raise_on_bottom env
         name ty m).
Proof.
  unfold Meet_env.add_concrete_equation_on_canonical_name. cbv zeta.
  rewrite kind_of_expanded_renamed.
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite find_equivariant.
  destruct (Typing_env.find (Meet_env.typing_env env) name
              (Some (Expanded_type.kind ty))) as [existing | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite of_non_alias_type_renamed.
  destruct (Expanded_type.of_non_alias_type existing) as [existing_ty | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite m_related.
  destruct (m env ty existing_ty) as [[res | res env'] | ];
    cbn [option_map rename_meet_result]; [ | | reflexivity ].
  - destruct raise_on_bottom; [ reflexivity | ].
    cbn [option_map or_bottom_map]. do 2 f_equal.
    apply map_typing_env_renamed. intros tenv.
    rewrite <- (proj1 (proj2 (rename_type_closed r))
                  (Expanded_type.kind ty)) at 1.
    apply replace_equation_equivariant.
  - destruct res as [ | | | ty']; cbn [rename_return_value];
      try reflexivity.
    + change (Meet_env.typing_env (rename_meet_env r env'))
        with (rename_typing_env r (Meet_env.typing_env env')).
      rewrite to_type_renamed, replace_concrete_equation_renamed.
      destruct (Meet_env.replace_concrete_equation (Meet_env.typing_env env')
                  name (Expanded_type.to_type ty));
        reflexivity.
    + change (Meet_env.typing_env (rename_meet_env r env'))
        with (rename_typing_env r (Meet_env.typing_env env')).
      rewrite to_type_renamed, replace_concrete_equation_renamed.
      destruct (Meet_env.replace_concrete_equation (Meet_env.typing_env env')
                  name (Expanded_type.to_type ty'));
        reflexivity.
Qed.

Lemma adding_equation_for_name_renamed (env : Meet_env.t)
    (name : name)
    (f f' : Meet_env.t -> option (Or_bottom.t Meet_env.t)) :
  (∀ env : Meet_env.t,
    f' (rename_meet_env r env)
    = option_map (or_bottom_map (rename_meet_env r)) (f env)) ->
  Meet_env.adding_equation_for_name (rename_meet_env r env)
    (rename_name r name) f'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.adding_equation_for_name env name f).
Proof.
  intros Hf. unfold Meet_env.adding_equation_for_name. cbv zeta.
  match goal with
  | |- match f' ?env' with _ => _ end = _ =>
    change env'
      with (rename_meet_env r
              {| Meet_env.typing_env := Meet_env.typing_env env;
                 Meet_env.adding_equations_for_names :=
                   name :: Meet_env.adding_equations_for_names env;
                 Meet_env.names_to_avoid := Meet_env.names_to_avoid env |})
  end.
  rewrite Hf.
  match goal with
  | |- match option_map _ ?result with _ => _ end = _ =>
    destruct result as [[env' | ] | ]
  end;
    reflexivity.
Qed.

Lemma add_or_delay_concrete_equation_on_canonical_name_renamed
    (raise_on_bottom : bool) (env : Meet_env.t) (name : name)
    (ty : Expanded_type.t) :
  Meet_env.add_or_delay_concrete_equation_on_canonical_name raise_on_bottom
    (rename_meet_env r env) (rename_name r name) (rename_expanded r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_or_delay_concrete_equation_on_canonical_name
         raise_on_bottom env name ty m).
Proof.
  unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name.
  change (Meet_env.adding_equations_for_names (rename_meet_env r env))
    with (map (rename_name r) (Meet_env.adding_equations_for_names env)).
  rewrite (among_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  destruct (Keyed.among name_eqb name
              (Meet_env.adding_equations_for_names env));
    [ reflexivity | ].
  apply adding_equation_for_name_renamed. intros env'.
  apply add_concrete_equation_on_canonical_name_renamed.
Qed.

Lemma add_concrete_equation_on_const_renamed
    (raise_on_bottom : bool) (env : Meet_env.t) (const : const)
    (ty : Expanded_type.t) :
  Meet_env.add_concrete_equation_on_const raise_on_bottom
    (rename_meet_env r env) const (rename_expanded r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_concrete_equation_on_const raise_on_bottom env const ty
         m).
Proof.
  unfold Meet_env.add_concrete_equation_on_const.
  rewrite <- (create_const_renamed r const) at 1. rewrite m_related.
  destruct (m env ty (Expanded_type.create_const const))
    as [[res | res env'] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  destruct raise_on_bottom; reflexivity.
Qed.

Lemma add_or_delay_concrete_equation_on_canonical_renamed
    (raise_on_bottom : bool) (env : Meet_env.t) (simple : simple)
    (ty : Expanded_type.t) :
  Meet_env.add_or_delay_concrete_equation_on_canonical raise_on_bottom
    (rename_meet_env r env) (rename_simple r simple) (rename_expanded r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_or_delay_concrete_equation_on_canonical raise_on_bottom
         env simple ty m).
Proof.
  destruct simple as [x | sym | c];
    cbn [Meet_env.add_or_delay_concrete_equation_on_canonical rename_simple].
  - exact
      (add_or_delay_concrete_equation_on_canonical_name_renamed
         raise_on_bottom env (Name_var x) ty).
  - exact
      (add_or_delay_concrete_equation_on_canonical_name_renamed
         raise_on_bottom env (Name_symbol sym) ty).
  - apply add_concrete_equation_on_const_renamed.
Qed.

Lemma add_concrete_equation_on_simple_renamed
    (raise_on_bottom : bool) (env : Meet_env.t) (simple : simple)
    (ty : Expanded_type.t) :
  Meet_env.add_concrete_equation_on_simple raise_on_bottom
    (rename_meet_env r env) (rename_simple r simple) (rename_expanded r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_concrete_equation_on_simple raise_on_bottom env simple ty
         m).
Proof.
  unfold Meet_env.add_concrete_equation_on_simple. cbv zeta.
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite get_canonical_ignoring_name_mode_equivariant.
  apply add_or_delay_concrete_equation_on_canonical_renamed.
Qed.

Lemma record_demotion_renamed (raise_on_bottom : bool)
    (env : Meet_env.t) (kind : flambda_kind) (demoted : name)
    (canonical : simple) :
  Meet_env.record_demotion raise_on_bottom (rename_meet_env r env) kind
    (rename_name r demoted) (rename_simple r canonical) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.record_demotion raise_on_bottom env kind demoted canonical m).
Proof.
  unfold Meet_env.record_demotion. cbv zeta.
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite find_equivariant.
  destruct (Typing_env.find (Meet_env.typing_env env) demoted (Some kind))
    as [ty_of_demoted | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite
    (map_typing_env_renamed r env
       (fun tenv =>
          Typing_env.replace_equation tenv demoted
            (alias_type_of kind canonical)));
    [ | intros tenv; rewrite <- (proj1 (rename_type_alias r));
        apply replace_equation_equivariant ].
  match goal with
  | |- context [Meet_env.typing_env (rename_meet_env r ?env')] =>
    change (Meet_env.typing_env (rename_meet_env r env'))
      with (rename_typing_env r (Meet_env.typing_env env'))
  end.
  rewrite <- rename_simple_of_name.
  change (Some (rename_simple r (simple_of_name demoted)))
    with (option_map (rename_simple r) (Some (simple_of_name demoted))).
  rewrite expand_head0_renamed.
  match goal with
  | |- match option_map _ ?expanded with _ => _ end = _ =>
    destruct expanded as [ty | ]
  end;
    cbn [option_map]; [ | reflexivity ].
  apply add_or_delay_concrete_equation_on_canonical_renamed.
Qed.

Lemma add_alias_between_canonicals_renamed
    (raise_on_bottom : bool) (env : Meet_env.t) (kind : flambda_kind)
    (canonical_element1 canonical_element2 : simple) :
  Meet_env.add_alias_between_canonicals raise_on_bottom
    (rename_meet_env r env) kind (rename_simple r canonical_element1)
    (rename_simple r canonical_element2) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_alias_between_canonicals raise_on_bottom env kind
         canonical_element1 canonical_element2 m).
Proof.
  unfold Meet_env.add_alias_between_canonicals.
  rewrite simple_equal_renamed.
  destruct (simple_equal canonical_element1 canonical_element2);
    [ reflexivity | ].
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite add_alias_equivariant.
  destruct (Typing_env.add_alias (Meet_env.typing_env env) canonical_element1
              canonical_element2) as [[ | result | ] | ];
    cbn [option_map or_unknown_or_bottom_map]; try reflexivity.
  - cbn [rename_add_alias_result Typing_env.demoted_name
         Typing_env.canonical_element Typing_env.add_alias_result_t].
    rewrite with_typing_env_renamed. apply record_demotion_renamed.
  - destruct raise_on_bottom; reflexivity.
Qed.

Lemma add_equation_on_simple0_renamed (raise_on_bottom : bool)
    (env : Meet_env.t) (simple : simple) (ty : flambda_type) :
  Meet_env.add_equation_on_simple0 raise_on_bottom (rename_meet_env r env)
    (rename_simple r simple) (rename_type r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_equation_on_simple0 raise_on_bottom env simple ty m).
Proof.
  unfold Meet_env.add_equation_on_simple0. cbv zeta.
  rewrite (proj2 (rename_type_alias r)), kind_rename_type.
  destruct (get_alias_opt ty) as [alias | ]; cbn [option_map].
  - change (Meet_env.typing_env (rename_meet_env r env))
      with (rename_typing_env r (Meet_env.typing_env env)).
    rewrite !get_canonical_ignoring_name_mode_equivariant.
    apply add_alias_between_canonicals_renamed.
  - rewrite of_non_alias_type_renamed.
    destruct (Expanded_type.of_non_alias_type ty) as [expanded | ];
      cbn [option_map]; [ | reflexivity ].
    apply add_concrete_equation_on_simple_renamed.
Qed.

Lemma add_equation0_renamed (raise_on_bottom : bool)
    (env : Meet_env.t) (name : name) (ty : flambda_type) :
  Meet_env.add_equation0 raise_on_bottom (rename_meet_env r env)
    (rename_name r name) (rename_type r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_equation0 raise_on_bottom env name ty m).
Proof.
  unfold Meet_env.add_equation0. rewrite <- rename_simple_of_name.
  apply add_equation_on_simple0_renamed.
Qed.

Lemma add_equations0_renamed (raise_on_bottom : bool)
    (equations : list (name * flambda_type)) :
  ∀ env : Meet_env.t,
    Meet_env.add_equations0 raise_on_bottom (rename_meet_env r env)
      (rename_equations r equations) m'
    = option_map (or_bottom_map (rename_meet_env r))
        (Meet_env.add_equations0 raise_on_bottom env equations m).
Proof.
  induction equations as [ | [name ty] equations IH]; intros env;
    cbn [rename_equations map Meet_env.add_equations0 fst snd];
    [ reflexivity | ].
  rewrite add_equation0_renamed.
  destruct (Meet_env.add_equation0 raise_on_bottom env name ty m)
    as [[env' | ] | ];
    cbn [option_map or_bottom_map]; [ apply IH | reflexivity | reflexivity ].
Qed.

Lemma add_env_extension0_renamed (raise_on_bottom : bool)
    (env : Meet_env.t) (extension : env_extension) :
  Meet_env.add_env_extension0 raise_on_bottom (rename_meet_env r env)
    (rename_env_extension r extension) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_env_extension0 raise_on_bottom env extension m).
Proof.
  destruct extension as [equations].
  cbn [Meet_env.add_env_extension0 rename_env_extension].
  change
    (map
       (fun equation =>
          (rename_name r (fst equation), rename_type r (snd equation)))
       equations)
    with (Keyed.map_bindings (rename_name r) (rename_type r) equations).
  rewrite
    (compact_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  apply add_equations0_renamed.
Qed.

Lemma add_equation_strict_renamed (env : Meet_env.t)
    (name : name) (ty : flambda_type) :
  Meet_env.add_equation_strict (rename_meet_env r env) (rename_name r name)
    (rename_type r ty) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_equation_strict env name ty m).
Proof.
  unfold Meet_env.add_equation_strict.
  change (Typing_env.is_bottom (Meet_env.typing_env (rename_meet_env r env)))
    with (Typing_env.is_bottom (Meet_env.typing_env env)).
  destruct (Typing_env.is_bottom (Meet_env.typing_env env));
    [ reflexivity | apply add_equation0_renamed ].
Qed.

Lemma add_env_extension_strict_renamed (env : Meet_env.t)
    (extension : env_extension) :
  Meet_env.add_env_extension_strict (rename_meet_env r env)
    (rename_env_extension r extension) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_env_extension_strict env extension m).
Proof.
  unfold Meet_env.add_env_extension_strict.
  change (Typing_env.is_bottom (Meet_env.typing_env (rename_meet_env r env)))
    with (Typing_env.is_bottom (Meet_env.typing_env env)).
  destruct (Typing_env.is_bottom (Meet_env.typing_env env));
    [ reflexivity | apply add_env_extension0_renamed ].
Qed.

Lemma bottom_on_exception_renamed (env : Meet_env.t)
    (result : option (Or_bottom.t Meet_env.t)) :
  Meet_env.bottom_on_exception (rename_meet_env r env)
    (option_map (or_bottom_map (rename_meet_env r)) result)
  = option_map (rename_meet_env r) (Meet_env.bottom_on_exception env result).
Proof.
  destruct result as [[env' | ] | ];
    cbn [Meet_env.bottom_on_exception option_map or_bottom_map];
    reflexivity.
Qed.

Lemma add_env_extension_maybe_bottom_renamed
    (env : Meet_env.t) (extension : env_extension) :
  Meet_env.add_env_extension_maybe_bottom (rename_meet_env r env)
    (rename_env_extension r extension) m'
  = option_map (rename_meet_env r)
      (Meet_env.add_env_extension_maybe_bottom env extension m).
Proof.
  unfold Meet_env.add_env_extension_maybe_bottom.
  rewrite add_env_extension0_renamed.
  destruct (Meet_env.add_env_extension0 false env extension m)
    as [[env' | ] | ];
    reflexivity.
Qed.

Lemma add_equation_renamed (env : Meet_env.t) (name : name)
    (ty : flambda_type) :
  Meet_env.add_equation (rename_meet_env r env) (rename_name r name)
    (rename_type r ty) m'
  = option_map (rename_meet_env r) (Meet_env.add_equation env name ty m).
Proof.
  unfold Meet_env.add_equation. rewrite add_equation0_renamed.
  apply bottom_on_exception_renamed.
Qed.

Lemma add_equation_on_simple_renamed (env : Meet_env.t)
    (simple : simple) (ty : flambda_type) :
  Meet_env.add_equation_on_simple (rename_meet_env r env)
    (rename_simple r simple) (rename_type r ty) m'
  = option_map (rename_meet_env r)
      (Meet_env.add_equation_on_simple env simple ty m).
Proof.
  unfold Meet_env.add_equation_on_simple.
  rewrite add_equation_on_simple0_renamed.
  apply bottom_on_exception_renamed.
Qed.

Lemma add_env_extension_renamed (env : Meet_env.t)
    (extension : env_extension) :
  Meet_env.add_env_extension (rename_meet_env r env)
    (rename_env_extension r extension) m'
  = option_map (rename_meet_env r)
      (Meet_env.add_env_extension env extension m).
Proof.
  unfold Meet_env.add_env_extension. rewrite add_env_extension0_renamed.
  apply bottom_on_exception_renamed.
Qed.

Lemma add_equations_on_params_renamed
    (params : list (variable * flambda_kind)) :
  ∀ (env : Meet_env.t) (param_types : list flambda_type),
    Meet_env.add_equations_on_params (rename_meet_env r env)
      (rename_params r params) (map (rename_type r) param_types) m'
    = option_map (rename_meet_env r)
        (Meet_env.add_equations_on_params env params param_types m).
Proof.
  induction params as [ | param params IH]; intros env [ | ty param_types];
    cbn [rename_params map Meet_env.add_equations_on_params fst snd];
    try reflexivity.
  change (Name_var (rename_variable r (fst param)))
    with (rename_name r (Name_var (fst param))).
  rewrite add_equation_renamed.
  destruct (Meet_env.add_equation env (Name_var (fst param)) ty m)
    as [env' | ];
    cbn [option_map]; [ apply IH | reflexivity ].
Qed.

Lemma add_alias0_renamed (raise_on_bottom : bool)
    (env : Meet_env.t) (simple1 simple2 : simple) :
  Meet_env.add_alias0 raise_on_bottom (rename_meet_env r env)
    (rename_simple r simple1) (rename_simple r simple2) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_alias0 raise_on_bottom env simple1 simple2 m).
Proof.
  unfold Meet_env.add_alias0. cbv zeta.
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite !get_canonical_ignoring_name_mode_equivariant,
    kind_of_simple_renamed.
  destruct
    (kind_of_simple (Meet_env.typing_env env)
       (Typing_env.get_canonical_simple_ignoring_name_mode
          (Meet_env.typing_env env) simple1))
    as [kind | ];
    [ | reflexivity ].
  apply add_alias_between_canonicals_renamed.
Qed.

Lemma add_alias_renamed (env : Meet_env.t)
    (simple1 simple2 : simple) :
  Meet_env.add_alias (rename_meet_env r env) (rename_simple r simple1)
    (rename_simple r simple2) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_alias env simple1 simple2 m).
Proof. apply add_alias0_renamed. Qed.

(* ======================================================================== *)
(* The meet of two types of which either may be an alias                    *)
(* ======================================================================== *)

Lemma meet_renamed (env : Meet_env.t) (t1 t2 : flambda_type) :
  Meet_env.meet (rename_meet_env r env) (rename_type r t1) (rename_type r t2)
    m'
  = option_map (rename_meet_result r (rename_expanded r))
      (Meet_env.meet env t1 t2 m).
Proof.
  unfold Meet_env.meet. cbv zeta. rewrite !kind_rename_type.
  destruct (negb (kind_eqb (kind t1) (kind t2))); [ reflexivity | ].
  change (Meet_env.typing_env (rename_meet_env r env))
    with (rename_typing_env r (Meet_env.typing_env env)).
  rewrite !get_alias_then_canonical_simple_exn_equivariant.
  destruct
    (Typing_env.get_alias_then_canonical_simple_exn (Meet_env.typing_env env)
       (Some Name_mode.in_types) None t1)
    as [simple1 | ];
    cbn [option_map]; [ | reflexivity ].
  destruct
    (Typing_env.get_alias_then_canonical_simple_exn (Meet_env.typing_env env)
       (Some Name_mode.in_types) None t2)
    as [simple2 | ];
    cbn [option_map]; [ | reflexivity ].
  destruct simple1 as [simple1 | ], simple2 as [simple2 | ];
    cbn [option_map].
  - rewrite add_alias_between_canonicals_renamed.
    destruct
      (Meet_env.add_alias_between_canonicals true env (kind t1) simple1
         simple2 m)
      as [[env' | ] | ];
      reflexivity.
  - change (@None simple) with (option_map (rename_simple r) None) at 1.
    rewrite expand_head0_renamed.
    destruct (expand_head0 (Meet_env.typing_env env) t2 None)
      as [expanded2 | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite add_or_delay_concrete_equation_on_canonical_renamed.
    destruct
      (Meet_env.add_or_delay_concrete_equation_on_canonical true env simple1
         expanded2 m)
      as [[env' | ] | ];
      reflexivity.
  - change (@None simple) with (option_map (rename_simple r) None) at 1.
    rewrite expand_head0_renamed.
    destruct (expand_head0 (Meet_env.typing_env env) t1 None)
      as [expanded1 | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite add_or_delay_concrete_equation_on_canonical_renamed.
    destruct
      (Meet_env.add_or_delay_concrete_equation_on_canonical true env simple2
         expanded1 m)
      as [[env' | ] | ];
      reflexivity.
  - change (@None simple) with (option_map (rename_simple r) None) at 1.
    rewrite expand_head0_renamed.
    destruct (expand_head0 (Meet_env.typing_env env) t1 None)
      as [expanded1 | ];
      cbn [option_map]; [ | reflexivity ].
    change (@None simple) with (option_map (rename_simple r) None) at 1.
    rewrite expand_head0_renamed.
    destruct (expand_head0 (Meet_env.typing_env env) t2 None)
      as [expanded2 | ];
      cbn [option_map]; [ | reflexivity ].
    apply m_related.
Qed.

Lemma meet_type_renamed (env : Meet_env.t)
    (t1 t2 : flambda_type) :
  Meet_env.meet_type (rename_meet_env r env) (rename_type r t1)
    (rename_type r t2) m'
  = option_map (rename_meet_result r (rename_type r))
      (Meet_env.meet_type env t1 t2 m).
Proof.
  unfold Meet_env.meet_type. rewrite meet_renamed.
  destruct (Meet_env.meet env t1 t2 m) as [result | ]; cbn [option_map];
    [ | reflexivity ].
  f_equal. apply map_result_renamed. intros expanded. apply to_type_renamed.
Qed.

End Meet_given.

(* ======================================================================== *)
(* A function run in an environment of the meet                             *)
(* ======================================================================== *)

Lemma use_meet_env_renamed (r : renaming) (tenv : Typing_env.t)
    (f f' : Meet_env.t -> option Meet_env.t) :
  (∀ env : Meet_env.t,
    f' (rename_meet_env r env) = option_map (rename_meet_env r) (f env)) ->
  Meet_env.use_meet_env (rename_typing_env r tenv) f'
  = option_map (rename_typing_env r) (Meet_env.use_meet_env tenv f).
Proof.
  intros Hf. unfold Meet_env.use_meet_env.
  rewrite create_renamed, Hf.
  destruct (f (Meet_env.create tenv)); reflexivity.
Qed.

Lemma use_meet_env_strict_renamed (r : renaming) (tenv : Typing_env.t)
    (f f' : Meet_env.t -> option Meet_env.t) :
  (∀ env : Meet_env.t,
    f' (rename_meet_env r env) = option_map (rename_meet_env r) (f env)) ->
  Meet_env.use_meet_env_strict (rename_typing_env r tenv) f'
  = option_map (or_bottom_map (rename_typing_env r))
      (Meet_env.use_meet_env_strict tenv f).
Proof.
  intros Hf. unfold Meet_env.use_meet_env_strict.
  change (Typing_env.is_bottom (rename_typing_env r tenv))
    with (Typing_env.is_bottom tenv).
  destruct (Typing_env.is_bottom tenv); [ reflexivity | ].
  rewrite create_renamed, Hf.
  destruct (f (Meet_env.create tenv)) as [env | ]; cbn [option_map];
    [ | reflexivity ].
  cbv zeta.
  change
    (Typing_env.is_bottom
       (Meet_env.final_typing_env (rename_meet_env r env)))
    with (Typing_env.is_bottom (Meet_env.final_typing_env env)).
  destruct (Typing_env.is_bottom (Meet_env.final_typing_env env));
    reflexivity.
Qed.

(* ======================================================================== *)
(* The meet as it is exposed, and adding to a typing environment            *)
(* ======================================================================== *)

(* [Type_grammar.is_null], [is_int_for_scrutinee] and
   [get_tag_for_block], for a scrutinee that may be a constant. *)

Lemma is_null_of_simple_renamed (r : renaming) (scrutinee : simple) :
  Flambda2_types.is_null_of_simple (rename_simple r scrutinee)
  = rename_type r (Flambda2_types.is_null_of_simple scrutinee).
Proof.
  destruct scrutinee as [x | sym | c];
    cbn [Flambda2_types.is_null_of_simple rename_simple].
  - symmetry. exact (proj2 (proj2 (rename_type_of_relation r (Name_var x)))).
  - symmetry.
    exact (proj2 (proj2 (rename_type_of_relation r (Name_symbol sym)))).
  - destruct c; reflexivity.
Qed.

Lemma is_int_of_simple_renamed (r : renaming) (scrutinee : simple) :
  Flambda2_types.is_int_of_simple (rename_simple r scrutinee)
  = rename_type r (Flambda2_types.is_int_of_simple scrutinee).
Proof.
  destruct scrutinee as [x | sym | c];
    cbn [Flambda2_types.is_int_of_simple rename_simple].
  - symmetry. exact (proj1 (rename_type_of_relation r (Name_var x))).
  - symmetry. exact (proj1 (rename_type_of_relation r (Name_symbol sym))).
  - reflexivity.
Qed.

Lemma get_tag_of_simple_renamed (r : renaming) (block : simple) :
  Flambda2_types.get_tag_of_simple (rename_simple r block)
  = rename_type r (Flambda2_types.get_tag_of_simple block).
Proof.
  destruct block as [x | sym | c];
    cbn [Flambda2_types.get_tag_of_simple rename_simple].
  - symmetry.
    exact (proj1 (proj2 (rename_type_of_relation r (Name_var x)))).
  - symmetry.
    exact (proj1 (proj2 (rename_type_of_relation r (Name_symbol sym)))).
  - reflexivity.
Qed.

(* The functions of the modules [Meet] and [Flambda2_types] of
   FlambdaMeet.v, for one renaming, two joins, choices and a fuel at
   which the renaming relates the two meets of expanded types: the
   functions with the second join, on the renamed arguments, give the
   renamed answer of those with the first. *)
Section Adding.

Variable r : renaming.
Variables join join' : join_scoped_envs.
Variable choices : meet_choices.
Variable fuel : nat.
Hypothesis meet_related :
  meet_expanded_head_related r (Meet.meet_expanded_head join choices fuel)
    (Meet.meet_expanded_head join' choices fuel).

(* [Meet.meet]. *)
Lemma exposed_meet_renamed :
  meet_function_related r (Meet.meet join choices fuel)
    (Meet.meet join' choices fuel).
Proof.
  intros tenv t1 t2. unfold Meet.meet.
  change (Typing_env.is_bottom (rename_typing_env r tenv))
    with (Typing_env.is_bottom tenv).
  destruct (Typing_env.is_bottom tenv); [ reflexivity | ].
  rewrite create_renamed, (meet_type_renamed r _ _ meet_related).
  destruct
    (Meet_env.meet_type (Meet_env.create tenv) t1 t2
       (Meet.meet_expanded_head join choices fuel))
    as [[res | res env] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  cbv zeta. rewrite extract_value_renamed, is_obviously_bottom_renamed.
  destruct (is_obviously_bottom (extract_value res t1 t2));
    [ reflexivity | ].
  rewrite final_typing_env_strict_renamed.
  destruct (Meet_env.final_typing_env_strict env) as [tenv' | ];
    reflexivity.
Qed.

(* [Meet.meet_shape]. *)
Lemma meet_shape_renamed (tenv : Typing_env.t) (t shape : flambda_type) :
  Meet.meet_shape join' choices fuel (rename_typing_env r tenv)
    (rename_type r t) (rename_type r shape)
  = option_map (or_bottom_map (rename_typing_env r))
      (Meet.meet_shape join choices fuel tenv t shape).
Proof.
  unfold Meet.meet_shape.
  change (Typing_env.is_bottom (rename_typing_env r tenv))
    with (Typing_env.is_bottom tenv).
  destruct (Typing_env.is_bottom tenv); [ reflexivity | ].
  rewrite exposed_meet_renamed.
  destruct (Meet.meet join choices fuel tenv t shape)
    as [[[ty tenv'] | ] | ];
    reflexivity.
Qed.

(* [Flambda2_types.add_equation]. *)
Lemma typing_env_add_equation_renamed (tenv : Typing_env.t) (name : name)
    (ty : flambda_type) :
  Flambda2_types.add_equation join' choices fuel (rename_typing_env r tenv)
    (rename_name r name) (rename_type r ty)
  = option_map (rename_typing_env r)
      (Flambda2_types.add_equation join choices fuel tenv name ty).
Proof.
  apply use_meet_env_renamed. intros env.
  apply (add_equation_renamed r _ _ meet_related).
Qed.

(* [Flambda2_types.add_equations_on_params]. *)
Lemma typing_env_add_equations_on_params_renamed (tenv : Typing_env.t)
    (params : list (variable * flambda_kind))
    (param_types : list flambda_type) :
  Flambda2_types.add_equations_on_params join' choices fuel
    (rename_typing_env r tenv) (rename_params r params)
    (map (rename_type r) param_types)
  = option_map (rename_typing_env r)
      (Flambda2_types.add_equations_on_params join choices fuel tenv params
         param_types).
Proof.
  apply use_meet_env_renamed. intros env.
  apply (add_equations_on_params_renamed r _ _ meet_related).
Qed.

(* [Flambda2_types.add_env_extension]. *)
Lemma typing_env_add_env_extension_renamed (tenv : Typing_env.t)
    (extension : env_extension) :
  Flambda2_types.add_env_extension join' choices fuel
    (rename_typing_env r tenv) (rename_env_extension r extension)
  = option_map (rename_typing_env r)
      (Flambda2_types.add_env_extension join choices fuel tenv extension).
Proof.
  apply use_meet_env_renamed. intros env.
  apply (add_env_extension_renamed r _ _ meet_related).
Qed.

(* The head that the scrutinee is given is a function of a variable:
   on the renamed variable it is to be the renamed head. *)
Lemma add_relation_renamed (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) (name_ty : flambda_type)
    (scrutinee_head scrutinee_head' : variable -> head_of_kind_value) :
  (∀ var : variable,
    scrutinee_head' (rename_variable r var)
    = rename_head_of_kind_value r (scrutinee_head var)) ->
  Flambda2_types.add_relation join' choices fuel (rename_typing_env r tenv)
    (rename_name r name) (rename_simple r scrutinee) (rename_type r name_ty)
    scrutinee_head'
  = option_map (rename_typing_env r)
      (Flambda2_types.add_relation join choices fuel tenv name scrutinee
         name_ty scrutinee_head).
Proof.
  intros Hhead. apply use_meet_env_renamed. intros env.
  rewrite (add_equation_renamed r _ _ meet_related).
  destruct
    (Meet_env.add_equation env name name_ty
       (Meet.meet_expanded_head join choices fuel))
    as [env' | ];
    cbn [option_map]; [ | reflexivity ].
  destruct name as [var | sym]; cbn [rename_name]; [ | reflexivity ].
  rewrite Hhead.
  exact
    (add_equation_on_simple_renamed r _ _ meet_related env' scrutinee
       (Type_value (Ok (No_alias (scrutinee_head var))))).
Qed.

(* [Flambda2_types.add_is_null_relation]. *)
Lemma add_is_null_relation_renamed (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) :
  Flambda2_types.add_is_null_relation join' choices fuel
    (rename_typing_env r tenv) (rename_name r name)
    (rename_simple r scrutinee)
  = option_map (rename_typing_env r)
      (Flambda2_types.add_is_null_relation join choices fuel tenv name
         scrutinee).
Proof.
  unfold Flambda2_types.add_is_null_relation.
  rewrite is_null_of_simple_renamed. apply add_relation_renamed.
  intros var. reflexivity.
Qed.

(* [Flambda2_types.add_is_int_relation]. *)
Lemma add_is_int_relation_renamed (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) :
  Flambda2_types.add_is_int_relation join' choices fuel
    (rename_typing_env r tenv) (rename_name r name)
    (rename_simple r scrutinee)
  = option_map (rename_typing_env r)
      (Flambda2_types.add_is_int_relation join choices fuel tenv name
         scrutinee).
Proof.
  unfold Flambda2_types.add_is_int_relation.
  rewrite is_int_of_simple_renamed. apply add_relation_renamed.
  intros var. reflexivity.
Qed.

(* [Flambda2_types.add_get_tag_relation]. *)
Lemma add_get_tag_relation_renamed (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) :
  Flambda2_types.add_get_tag_relation join' choices fuel
    (rename_typing_env r tenv) (rename_name r name)
    (rename_simple r scrutinee)
  = option_map (rename_typing_env r)
      (Flambda2_types.add_get_tag_relation join choices fuel tenv name
         scrutinee).
Proof.
  unfold Flambda2_types.add_get_tag_relation.
  rewrite get_tag_of_simple_renamed. apply add_relation_renamed.
  intros var. reflexivity.
Qed.

End Adding.
