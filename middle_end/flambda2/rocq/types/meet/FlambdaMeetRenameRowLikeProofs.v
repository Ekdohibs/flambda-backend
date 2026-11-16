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


(* For the proofs of the statements of FlambdaMeetRenameSpec.v: the
   meet of two row-like structures under a renaming. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv
  FlambdaEnvRenameSpec FlambdaEnvRenameProofs FlambdaMeetGrammar
  FlambdaExpandHead FlambdaProversRenameSpec FlambdaProversRenameProofs
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetRenameSpec
  FlambdaEnvRenameMeetProofs FlambdaMeetRenameScopesProofs
  FlambdaMeetRowLikeProofs.
Import ListNotations.

(* ======================================================================== *)
(* The meet of two row-like structures                                      *)
(* ======================================================================== *)

Definition rename_extra (r : renaming) (extra : list (variable * flambda_kind))
    : list (variable * flambda_kind) :=
  Keyed.map_keys (rename_variable r) extra.

Lemma union_extra_variables_renamed (r : renaming)
    (vars : list (variable * flambda_kind)) :
  ∀ extra : list (variable * flambda_kind),
    union_extra_variables (rename_extra r extra) (rename_extra r vars)
    = option_map (rename_extra r) (union_extra_variables extra vars).
Proof.
  induction vars as [ | [var kind] vars IH]; intros extra; [ reflexivity | ].
  simpl. unfold rename_extra at 1, Keyed.map_keys at 1.
  rewrite (lookup_variable_renamed r (fun a : flambda_kind => a) var extra).
  destruct (lookup var extra) as [kind' | ]; simpl.
  - destruct (kind_eqb kind kind'); [ apply IH | reflexivity ].
  - rewrite <- IH. f_equal. unfold rename_extra, Keyed.map_keys,
      Keyed.map_bindings. rewrite map_app. reflexivity.
Qed.

Lemma rename_extra_rev (r : renaming)
    (extra : list (variable * flambda_kind)) :
  rev (rename_extra r extra) = rename_extra r (rev extra).
Proof.
  unfold rename_extra, Keyed.map_keys, Keyed.map_bindings.
  symmetry. apply map_rev.
Qed.

Lemma defined_variables_with_kinds_renamed (r : renaming)
    (level : Typing_env_level.t) :
  Typing_env_level.defined_variables_with_kinds (rename_level r level)
  = rename_extra r (Typing_env_level.defined_variables_with_kinds level).
Proof. reflexivity. Qed.

Lemma define_extra_variables_renamed (choices : meet_choices) (r : renaming)
    (extra : list (variable * flambda_kind)) :
  ∀ env : option Meet_env.t,
    List.fold_left
      (fun env binding =>
         let* env := env in
         define_in_types choices env (fst binding) (snd binding))
      (rename_extra r extra) (option_map (rename_meet_env r) env)
    = option_map (rename_meet_env r)
        (List.fold_left
           (fun env binding =>
              let* env := env in
              define_in_types choices env (fst binding) (snd binding))
           extra env).
Proof.
  induction extra as [ | [var kind] extra IH]; intros env; [ reflexivity | ].
  simpl. rewrite <- IH. f_equal.
  destruct env as [env | ]; [ | reflexivity ].
  exact (define_in_types_renamed choices r env var kind).
Qed.

(* A case, the state of the traversal, and what a step of it answers,
   renamed. The index of a case holds no name. *)
Definition rename_case {lattice shape maps_to : Type}
    (rm : renaming -> maps_to -> maps_to) (r : renaming)
    (c : row_like_case lattice shape maps_to env_extension)
    : row_like_case lattice shape maps_to env_extension :=
  Row_like_case (rm r (case_maps_to c)) (case_index c)
    (rename_env_extension r (case_env_extension c)).

Definition rename_row_like_state (r : renaming) (state : row_like_state)
    : row_like_state :=
  {| extra_variables := rename_extra r (extra_variables state);
     result_env := option_map (map (rename_typing_env r)) (result_env state);
     result_is_t1 := result_is_t1 state;
     result_is_t2 := result_is_t2 state;
     avoid := map (rename_variable r) (avoid state) |}.

Section Row_like.

Variable r : renaming.
Variables join join' : join_scoped_envs.
Variable choices : meet_choices.
Variables m m' : meet_expanded_head.

Hypothesis Hjoin : join_related_at r join join' m m'.
Hypothesis Hm : meet_expanded_head_related r m m'.

Context {lattice shape maps_to row_tag known : Type}.

Let case : Type := row_like_case lattice shape maps_to env_extension.

Variable rm : renaming -> maps_to -> maps_to.
Variable rk : renaming -> known -> known.

Let rename_step (r : renaming)
    (res : option (Or_unknown.t case) * row_like_state)
    : option (Or_unknown.t case) * row_like_state :=
  (option_map (or_unknown_map (rename_case rm r)) (fst res),
   rename_row_like_state r (snd res)).

Variables meet_maps_to meet_maps_to' :
  Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to).
Variable equal_index : lattice -> lattice -> bool.
Variable subset_index : lattice -> lattice -> bool.
Variable union_index : lattice -> lattice -> lattice.
Variable meet_shape : shape -> shape -> Or_bottom.t shape.
Variable is_empty_map_known : known -> bool.
Variable get_singleton_map_known : known -> option (row_tag * case).
Variable merge_map_known :
  (row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
   row_like_state -> option (option (Or_unknown.t case) * row_like_state)) ->
  known -> known -> row_like_state -> option (known * row_like_state).

Hypothesis Hmaps :
  meet_of_parts_related r (rm r) (rm r) meet_maps_to meet_maps_to'.
Hypothesis Hempty :
  ∀ k : known, is_empty_map_known (rk r k) = is_empty_map_known k.
Hypothesis Hsingleton :
  ∀ k : known,
    get_singleton_map_known (rk r k)
    = option_map (fun tc => (fst tc, rename_case rm r (snd tc)))
        (get_singleton_map_known k).
Hypothesis Hmerge :
  ∀ (f f' :
      row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
      row_like_state -> option (option (Or_unknown.t case) * row_like_state))
    (k1 k2 : known) (state : row_like_state),
    (∀ (tag : row_tag) (c1 c2 : option (Or_unknown.t case))
       (state : row_like_state),
      f' tag (option_map (or_unknown_map (rename_case rm r)) c1)
        (option_map (or_unknown_map (rename_case rm r)) c2)
        (rename_row_like_state r state)
      = option_map (rename_step r) (f tag c1 c2 state)) ->
    merge_map_known f' (rk r k1) (rk r k2) (rename_row_like_state r state)
    = option_map
        (fun res => (rk r (fst res), rename_row_like_state r (snd res)))
        (merge_map_known f k1 k2 state).

(* The parts of [meet_row_like] (FlambdaMeetRowLikeProofs.v), one by
   one. *)

Lemma not_t1_renamed (state : row_like_state) :
  not_t1 (rename_row_like_state r state)
  = rename_row_like_state r (not_t1 state).
Proof. reflexivity. Qed.

Lemma not_t2_renamed (state : row_like_state) :
  not_t2 (rename_row_like_state r state)
  = rename_row_like_state r (not_t2 state).
Proof. reflexivity. Qed.

Lemma update_refs_renamed {A : Type}
    (res : meet_return_value A) (state : row_like_state) :
  update_refs res (rename_row_like_state r state)
  = rename_row_like_state r (update_refs res state).
Proof. destruct res; reflexivity. Qed.

Lemma update_refs_return_value {A : Type} (f : A -> A)
    (res : meet_return_value A) (state : row_like_state) :
  update_refs (rename_return_value f res) state = update_refs res state.
Proof. destruct res; reflexivity. Qed.

Lemma row_add_extra_renamed (common_scope : Scope.t)
    (scoped_env : Typing_env.t) (state : row_like_state) :
  row_add_extra common_scope (rename_typing_env r scoped_env)
    (rename_row_like_state r state)
  = option_map
      (fun res =>
         (rename_env_extension r (fst res), rename_row_like_state r (snd res)))
      (row_add_extra common_scope scoped_env state).
Proof.
  unfold row_add_extra. rewrite cut_equivariant.
  destruct (Typing_env.cut scoped_env common_scope) as [level | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite defined_variables_with_kinds_renamed.
  rewrite rename_extra_rev.
  change (extra_variables (rename_row_like_state r state))
    with (rename_extra r (extra_variables state)).
  rewrite union_extra_variables_renamed.
  change (@rev (positive * flambda_kind))
    with (@rev (variable * flambda_kind)).
  match goal with
  | |- context [option_map (rename_extra r) ?u] =>
    destruct u as [extra | ]; [ | reflexivity ]
  end.
  cbn [option_map fst snd]. rewrite extension_of_level_renamed.
  reflexivity.
Qed.

Lemma row_join_all_renamed (initial_env : Meet_env.t)
    (common_scope : Scope.t) (initial_tenv : Typing_env.t)
    (scoped_envs : list Typing_env.t) (state : row_like_state) :
  row_join_all join' choices m' (rename_meet_env r initial_env) common_scope
    (rename_typing_env r initial_tenv)
    (map (rename_typing_env r) scoped_envs) (rename_row_like_state r state)
  = option_map (rename_meet_env r)
      (row_join_all join choices m initial_env common_scope initial_tenv
         scoped_envs state).
Proof.
  unfold row_join_all.
  change (avoid (rename_row_like_state r state))
    with (map (rename_variable r) (avoid state)).
  change (extra_variables (rename_row_like_state r state))
    with (rename_extra r (extra_variables state)).
  rewrite with_names_to_avoid_renamed, (Hjoin _ _ _ _).
  destruct (join m common_scope
              (Meet_env.with_names_to_avoid initial_env (avoid state))
              initial_tenv scoped_envs) as [result_env | ]; [ | reflexivity ].
  cbn [option_map].
  exact (define_extra_variables_renamed choices r (extra_variables state)
           (Some result_env)).
Qed.

Lemma row_join_result_env_renamed (need_join : bool)
    (scoped_env : Typing_env.t) (names : list variable)
    (state : row_like_state) :
  row_join_result_env need_join (rename_typing_env r scoped_env)
    (map (rename_variable r) names) (rename_row_like_state r state)
  = option_map (rename_row_like_state r)
      (row_join_result_env need_join scoped_env names state).
Proof.
  unfold row_join_result_env.
  destruct state as [extra [envs | ] is1 is2 names']; destruct need_join;
    reflexivity.
Qed.

Lemma row_meet_index_renamed (env : Meet_env.t)
    (i1 i2 : row_like_index lattice shape) :
  row_meet_index equal_index subset_index union_index meet_shape
    (rename_meet_env r env) i1 i2
  = match
      row_meet_index equal_index subset_index union_index meet_shape env i1
        i2
    with
    | Meet_result.Bottom res => Meet_result.Bottom res
    | Meet_result.Ok res env => Meet_result.Ok res (rename_meet_env r env)
    end.
Proof.
  unfold row_meet_index.
  destruct (meet_shape (index_shape i1) (index_shape i2)); [ | reflexivity ].
  destruct (index_domain i1) as [d1 | d1], (index_domain i2) as [d2 | d2].
  all: repeat match goal with
              | |- context [if ?c then _ else _] =>
                match type of c with
                | bool => destruct c
                end
              end.
  all: reflexivity.
Qed.

Lemma row_add_case_extension_renamed (base_env : Meet_env.t)
    (extension : env_extension) (state : row_like_state) :
  row_add_case_extension m' (rename_meet_env r base_env)
    (rename_env_extension r extension) (rename_row_like_state r state)
  = option_map (or_bottom_map (rename_meet_env r))
      (row_add_case_extension m base_env extension state).
Proof.
  unfold row_add_case_extension.
  change (avoid (rename_row_like_state r state))
    with (map (rename_variable r) (avoid state)).
  rewrite with_names_to_avoid_renamed.
  rewrite (add_env_extension_strict_renamed r m m' Hm).
  destruct (Meet_env.add_env_extension_strict
              (Meet_env.with_names_to_avoid base_env (avoid state)) extension
              m) as [[env | ] | ]; reflexivity.
Qed.

Lemma env_extension_is_empty_renamed {A : Type}
    (extension : env_extension) (a b : A) :
  match rename_env_extension r extension with
  | Env_extension [] => a
  | Env_extension (_ :: _) => b
  end
  = match extension with
    | Env_extension [] => a
    | Env_extension (_ :: _) => b
    end.
Proof. destruct extension as [[ | binding bindings]]; reflexivity. Qed.

Lemma row_case_result_renamed (common_scope : Scope.t)
    (need_join : bool) (case1 case2 : case)
    (index_result : meet_return_value (row_like_index lattice shape))
    (maps_to_result : meet_return_value maps_to) (env : Meet_env.t)
    (state : row_like_state) :
  row_case_result choices common_scope need_join (rename_case rm r case1)
    (rename_case rm r case2) index_result
    (rename_return_value (rm r) maps_to_result) (rename_meet_env r env)
    (rename_row_like_state r state)
  = option_map (rename_step r)
      (row_case_result choices common_scope need_join case1 case2
         index_result maps_to_result env state).
Proof.
  unfold row_case_result. unfold Meet_env.final_typing_env.
  rewrite typing_env_renamed, names_to_avoid_renamed.
  rewrite row_join_result_env_renamed.
  destruct (row_join_result_env need_join (Meet_env.typing_env env)
              (Meet_env.names_to_avoid env) state) as [state' | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite update_refs_return_value.
  rewrite !update_refs_renamed.
  set (state'' := update_refs maps_to_result (update_refs index_result state')).
  change (case_maps_to (rename_case rm r case1))
    with (rm r (case_maps_to case1)).
  change (case_maps_to (rename_case rm r case2))
    with (rm r (case_maps_to case2)).
  rewrite extract_value_renamed.
  change (case_index (rename_case rm r case1)) with (case_index case1).
  change (case_index (rename_case rm r case2)) with (case_index case2).
  rewrite !row_add_extra_renamed.
  destruct need_join.
  - destruct (row_add_extra common_scope (Meet_env.typing_env env) state'')
      as [[extension state3] | ]; [ | reflexivity ].
    cbn [option_map fst snd].
    rewrite (env_extension_is_empty_renamed extension).
    destruct extension as [[ | binding bindings]]; reflexivity.
  - destruct (lift_variables_of_single_case choices); [ | reflexivity ].
    destruct (row_add_extra common_scope (Meet_env.typing_env env) state'')
      as [[extension state3] | ]; reflexivity.
Qed.

Lemma row_meet_case_renamed (common_scope : Scope.t)
    (need_join : bool) (env : Meet_env.t) (case1 case2 : case)
    (state : row_like_state) :
  row_meet_case choices m' meet_maps_to' equal_index subset_index union_index
    meet_shape common_scope need_join (rename_meet_env r env)
    (rename_case rm r case1) (rename_case rm r case2)
    (rename_row_like_state r state)
  = option_map (rename_step r)
      (row_meet_case choices m meet_maps_to equal_index subset_index
         union_index meet_shape common_scope need_join env case1 case2 state).
Proof.
  unfold row_meet_case.
  change (avoid (rename_row_like_state r state))
    with (map (rename_variable r) (avoid state)).
  rewrite with_names_to_avoid_renamed.
  change (case_index (rename_case rm r case1)) with (case_index case1).
  change (case_index (rename_case rm r case2)) with (case_index case2).
  rewrite row_meet_index_renamed.
  destruct (row_meet_index equal_index subset_index union_index meet_shape
              (Meet_env.with_names_to_avoid env (avoid state))
              (case_index case1) (case_index case2))
    as [res | index_result env1].
  { cbn [option_map]. rewrite update_refs_renamed. reflexivity. }
  change (case_maps_to (rename_case rm r case1))
    with (rm r (case_maps_to case1)).
  change (case_maps_to (rename_case rm r case2))
    with (rm r (case_maps_to case2)).
  rewrite Hmaps.
  destruct (meet_maps_to env1 (case_maps_to case1) (case_maps_to case2))
    as [[res | maps_to_result env2] | ]; [ | | reflexivity ].
  { cbn [rename_meet_result option_map]. rewrite update_refs_renamed.
    reflexivity. }
  cbn [rename_meet_result option_map].
  change (case_env_extension (rename_case rm r case1))
    with (rename_env_extension r (case_env_extension case1)).
  change (case_env_extension (rename_case rm r case2))
    with (rename_env_extension r (case_env_extension case2)).
  rewrite (add_env_extension_strict_renamed r m m' Hm).
  destruct (Meet_env.add_env_extension_strict env2 (case_env_extension case1)
              m) as [[env3 | ] | ]; [ | | reflexivity ].
  2: { cbn [option_map or_bottom_map]. rewrite update_refs_renamed.
       reflexivity. }
  cbn [option_map or_bottom_map].
  rewrite (add_env_extension_strict_renamed r m m' Hm).
  destruct (Meet_env.add_env_extension_strict env3 (case_env_extension case2)
              m) as [[env4 | ] | ]; [ | | reflexivity ].
  2: { cbn [option_map or_bottom_map]. rewrite update_refs_renamed.
       reflexivity. }
  cbn [option_map or_bottom_map Meet_env.final_typing_env_strict].
  apply row_case_result_renamed.
Qed.

Lemma row_meet_case_with_unknown_renamed (need_join : bool)
    (base_env : Meet_env.t) (other_case : case)
    (change : row_like_state -> row_like_state) (state : row_like_state) :
  (∀ state : row_like_state,
    change (rename_row_like_state r state)
    = rename_row_like_state r (change state)) ->
  row_meet_case_with_unknown m' need_join (rename_meet_env r base_env)
    (rename_case rm r other_case) change (rename_row_like_state r state)
  = option_map (rename_step r)
      (row_meet_case_with_unknown m need_join base_env other_case change
         state).
Proof.
  intros Hchange. unfold row_meet_case_with_unknown.
  change (case_env_extension (rename_case rm r other_case))
    with (rename_env_extension r (case_env_extension other_case)).
  rewrite row_add_case_extension_renamed.
  destruct (row_add_case_extension m base_env
              (case_env_extension other_case) state) as [[env | ] | ];
    [ | reflexivity | reflexivity ].
  cbn [option_map or_bottom_map]. unfold Meet_env.final_typing_env.
  rewrite typing_env_renamed, names_to_avoid_renamed.
  rewrite row_join_result_env_renamed.
  destruct (row_join_result_env need_join (Meet_env.typing_env env)
              (Meet_env.names_to_avoid env) state) as [state' | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite Hchange. reflexivity.
Qed.

Lemma row_meet_knowns_renamed (common_scope : Scope.t)
    (need_join : bool) (base_env : Meet_env.t)
    (other1 other2 : Or_bottom.t case)
    (case1 case2 : option (Or_unknown.t case)) (state : row_like_state) :
  row_meet_knowns choices m' meet_maps_to' equal_index subset_index union_index
    meet_shape common_scope need_join (rename_meet_env r base_env)
    (or_bottom_map (rename_case rm r) other1)
    (or_bottom_map (rename_case rm r) other2)
    (option_map (or_unknown_map (rename_case rm r)) case1)
    (option_map (or_unknown_map (rename_case rm r)) case2)
    (rename_row_like_state r state)
  = option_map (rename_step r)
      (row_meet_knowns choices m meet_maps_to equal_index subset_index
         union_index meet_shape common_scope need_join base_env other1 other2
         case1 case2 state).
Proof.
  unfold row_meet_knowns.
  destruct case1 as [[case1 | ] | ], case2 as [[case2 | ] | ];
    cbn [option_map or_unknown_map].
  all: try (destruct other1 as [other1 | ]; cbn [or_bottom_map]).
  all: try (destruct other2 as [other2 | ]; cbn [or_bottom_map]).
  all: try reflexivity.
  all: try apply row_meet_case_renamed.
  all: try (apply row_meet_case_with_unknown_renamed; reflexivity).
  all: unfold Meet_env.final_typing_env; rewrite typing_env_renamed.
  all: change (avoid (rename_row_like_state r state))
         with (map (rename_variable r) (avoid state)).
  all: rewrite row_join_result_env_renamed.
  all: destruct (row_join_result_env need_join (Meet_env.typing_env base_env)
                   (avoid state) state); reflexivity.
Qed.

Lemma row_need_join_renamed (k1 k2 : known)
    (other1 other2 : Or_bottom.t case) :
  row_need_join is_empty_map_known get_singleton_map_known (rk r k1) (rk r k2)
    (or_bottom_map (rename_case rm r) other1)
    (or_bottom_map (rename_case rm r) other2)
  = row_need_join is_empty_map_known get_singleton_map_known k1 k2 other1
      other2.
Proof.
  unfold row_need_join. rewrite !Hsingleton, !Hempty.
  destruct other1, other2, (get_singleton_map_known k1),
    (get_singleton_map_known k2); reflexivity.
Qed.

Lemma row_initial_state_renamed (base_env : Meet_env.t) :
  row_initial_state (rename_meet_env r base_env)
  = rename_row_like_state r (row_initial_state base_env).
Proof. reflexivity. Qed.

Lemma row_match_with_input_renamed {A : Type} (f : A -> A)
    (state : row_like_state) (v : A) :
  row_match_with_input (rename_row_like_state r state) (f v)
  = rename_return_value f (row_match_with_input state v).
Proof.
  unfold row_match_with_input.
  change (result_is_t1 (rename_row_like_state r state))
    with (result_is_t1 state).
  change (result_is_t2 (rename_row_like_state r state))
    with (result_is_t2 state).
  destruct (result_is_t1 state), (result_is_t2 state); reflexivity.
Qed.

Lemma row_meet_others_renamed (common_scope : Scope.t)
    (need_join : bool) (base_env : Meet_env.t)
    (other1 other2 : Or_bottom.t case)
    (state : row_like_state) :
  row_meet_others choices m' meet_maps_to' equal_index subset_index union_index
    meet_shape common_scope need_join (rename_meet_env r base_env)
    (or_bottom_map (rename_case rm r) other1)
    (or_bottom_map (rename_case rm r) other2) (rename_row_like_state r state)
  = option_map
      (fun res =>
         (or_bottom_map (rename_case rm r) (fst res),
          rename_row_like_state r (snd res)))
      (row_meet_others choices m meet_maps_to equal_index subset_index
         union_index meet_shape common_scope need_join base_env other1 other2
         state).
Proof.
  unfold row_meet_others.
  destruct other1 as [other1 | ], other2 as [other2 | ];
    cbn [or_bottom_map]; try reflexivity.
  rewrite row_meet_case_renamed.
  destruct (row_meet_case choices m meet_maps_to equal_index subset_index
              union_index meet_shape common_scope need_join base_env other1
              other2 state) as [[[[c | ] | ] state'] | ]; reflexivity.
Qed.

Lemma row_finish_renamed (initial_env : Meet_env.t)
    (common_scope : Scope.t) (initial_tenv : Typing_env.t) (cases : known)
    (other : Or_bottom.t case) (state : row_like_state) :
  row_finish join' choices m' is_empty_map_known (rename_meet_env r initial_env)
    common_scope (rename_typing_env r initial_tenv) (rk r cases)
    (or_bottom_map (rename_case rm r) other) (rename_row_like_state r state)
  = option_map
      (rename_meet_result r
         (fun x => (rk r (fst x), or_bottom_map (rename_case rm r) (snd x))))
      (row_finish join choices m is_empty_map_known initial_env common_scope
         initial_tenv cases other state).
Proof.
  unfold row_finish. rewrite Hempty.
  set (cond := is_empty_map_known cases && _).
  replace (is_empty_map_known cases && _) with cond
    by (destruct other; reflexivity).
  destruct cond; [ reflexivity | ].
  change (result_env (rename_row_like_state r state))
    with (option_map (map (rename_typing_env r)) (result_env state)).
  destruct (result_env state) as [scoped_envs | ]; cbn [option_map].
  - rewrite row_join_all_renamed.
    destruct (row_join_all join choices m initial_env common_scope
                initial_tenv scoped_envs state) as [env | ]; [ | reflexivity ].
    cbn [option_map rename_meet_result].
    pose proof
      (row_match_with_input_renamed
         (fun x : known * Or_bottom.t case =>
            (rk r (fst x), or_bottom_map (rename_case rm r) (snd x)))
         state (cases, other))
      as Hinput.
    cbn [fst snd] in Hinput.
    exact (f_equal (fun v => Some (Meet_result.Ok v (rename_meet_env r env)))
             Hinput).
  - reflexivity.
Qed.

Lemma meet_row_like_renamed (env : Meet_env.t)
    (k1 k2 : known) (o1 o2 : Or_bottom.t case) :
  meet_row_like join' choices m' meet_maps_to' equal_index subset_index
    union_index meet_shape is_empty_map_known get_singleton_map_known
    merge_map_known (rename_meet_env r env) (rk r k1) (rk r k2)
    (or_bottom_map (rename_case rm r) o1)
    (or_bottom_map (rename_case rm r) o2)
  = option_map
      (rename_meet_result r
         (fun x => (rk r (fst x), or_bottom_map (rename_case rm r) (snd x))))
      (meet_row_like join choices m meet_maps_to equal_index subset_index
         union_index meet_shape is_empty_map_known get_singleton_map_known
         merge_map_known env k1 k2 o1 o2).
Proof.
  rewrite !meet_row_like_unfold. cbv zeta.
  rewrite enter_scope_renamed. cbn [snd].
  rewrite typing_env_renamed, current_scope_equivariant.
  rewrite row_need_join_renamed, row_initial_state_renamed.
  set (common_scope := Typing_env.current_scope (Meet_env.typing_env env)).
  set (base_env := snd (Meet_env.enter_scope env)).
  set (need_join :=
         row_need_join is_empty_map_known get_singleton_map_known k1 k2 o1 o2).
  rewrite
    (Hmerge
       (fun _ case1 case2 =>
          row_meet_knowns choices m meet_maps_to equal_index subset_index
            union_index meet_shape common_scope need_join base_env o1 o2 case1
            case2)).
  2: { intros tag c1 c2 state. apply row_meet_knowns_renamed. }
  destruct (merge_map_known _ k1 k2 (row_initial_state base_env))
    as [[cases state] | ]; [ | reflexivity ].
  cbn [option_map fst snd]. rewrite row_meet_others_renamed.
  destruct (row_meet_others choices m meet_maps_to equal_index subset_index
              union_index meet_shape common_scope need_join base_env o1 o2
              state) as [[other state'] | ]; [ | reflexivity ].
  cbn [option_map fst snd]. apply row_finish_renamed.
Qed.

End Row_like.
