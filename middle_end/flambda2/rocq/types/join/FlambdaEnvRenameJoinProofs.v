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


(* Proofs of the statements of FlambdaEnvRenameJoinSpec.v, and the
   functions of the join of environments on renamed arguments. Nothing
   here needs review.

   After the statements: for each function of FlambdaJoinEnv.v, in the
   order of that file, a lemma named after the function says that on
   renamed arguments it gives the renamed answer. A function that is
   given a function (the source of new names, the parameters, the
   function that imports a variable) is given, on the renamed side,
   one that is related to it. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeEqualSpec FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaTypingEnv FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaExpandHead FlambdaTypeProject
  FlambdaTypeProjectRenameProofs FlambdaJoinEnv FlambdaJoinEnvSpec
  FlambdaJoinTypesWithEnvSpec FlambdaMeet FlambdaJoinMeet
  FlambdaEnvRenameJoinSpec.
From OxCamlFlambda2Types Require FlambdaProversRenameSpec
  FlambdaProversRenameProofs FlambdaEnvRenameMeetPartsProofs
  FlambdaEnvRenameJoinTypesProofs.
Import ListNotations.

(* ======================================================================== *)
(* Lists and maps                                                           *)
(* ======================================================================== *)

Lemma map_rename_variable_inverse_l (r : renaming) (names : list variable) :
  map (rename_variable (inverse_renaming r)) (map (rename_variable r) names)
  = names.
Proof.
  rewrite map_map. rewrite <- (map_id names) at 2. apply map_ext.
  intros x. apply rename_variable_inverse_l.
Qed.

Lemma map_rename_variable_inverse_r (r : renaming) (names : list variable) :
  map (rename_variable r) (map (rename_variable (inverse_renaming r)) names)
  = names.
Proof.
  rewrite map_map. rewrite <- (map_id names) at 2. apply map_ext.
  intros x. apply rename_variable_inverse_r.
Qed.

(* A map whose keys are kept and whose bindings are renamed: the laws
   of a map whose keys are renamed, at the renaming of keys that moves
   none. *)
Section Kept_keys.

Context {K : Type} (eqb : K -> K -> bool).

Lemma is_bound_map_values {B C : Type} (g : B -> C) (k : K)
    (bindings : Keyed.finite_map K B) :
  Keyed.is_bound eqb k (Keyed.map_values g bindings)
  = Keyed.is_bound eqb k bindings.
Proof.
  exact (is_bound_renamed eqb (fun k => k) (fun _ _ => eq_refl) g k bindings).
Qed.

Lemma domain_map_values {B C : Type} (g : B -> C)
    (bindings : Keyed.finite_map K B) :
  Keyed.domain eqb (Keyed.map_values g bindings) = Keyed.domain eqb bindings.
Proof.
  etransitivity;
    [ exact (domain_renamed eqb (fun k => k) (fun _ _ => eq_refl) g bindings)
    | apply map_id ].
Qed.

Lemma compact_map_values {B C : Type} (g : B -> C)
    (bindings : Keyed.finite_map K B) :
  Keyed.compact eqb (Keyed.map_values g bindings)
  = Keyed.map_values g (Keyed.compact eqb bindings).
Proof.
  exact
    (compact_keys_renamed eqb (fun k => k) (fun _ _ => eq_refl) g bindings).
Qed.

End Kept_keys.

(* [for_all_bindings_b] of a map whose keys are renamed. *)
Section Renamed_keys.

Context {K : Type} (eqb : K -> K -> bool) (f : K -> K).
Hypothesis eqb_renamed : ∀ k k' : K, eqb (f k) (f k') = eqb k k'.

Lemma for_all_bindings_b_renamed {B C : Type} (g : B -> C)
    (p : K -> B -> bool) (p' : K -> C -> bool)
    (bindings : Keyed.finite_map K B) :
  (∀ (k : K) (b : B), p' (f k) (g b) = p k b) ->
  Keyed.for_all_bindings_b eqb p' (Keyed.map_bindings f g bindings)
  = Keyed.for_all_bindings_b eqb p bindings.
Proof.
  intros Hp. unfold Keyed.for_all_bindings_b, Keyed.map_bindings.
  change (@nil K) with (map f (@nil K)) at 1.
  generalize (@nil K) as seen.
  induction bindings as [ | [k b] bindings IH]; intros seen;
    cbn [map fst snd]; [ reflexivity | ].
  rewrite (among_keys_renamed eqb f eqb_renamed), Hp.
  change (f k :: map f seen) with (map f (k :: seen)). rewrite IH.
  reflexivity.
Qed.

End Renamed_keys.

Lemma for_all_bindings_b_map_values {K B C : Type} (eqb : K -> K -> bool)
    (g : B -> C) (p : K -> B -> bool) (p' : K -> C -> bool)
    (bindings : Keyed.finite_map K B) :
  (∀ (k : K) (b : B), p' k (g b) = p k b) ->
  Keyed.for_all_bindings_b eqb p' (Keyed.map_values g bindings)
  = Keyed.for_all_bindings_b eqb p bindings.
Proof.
  intros Hp.
  exact
    (for_all_bindings_b_renamed eqb (fun k => k) (fun _ _ => eq_refl) g p p'
       bindings Hp).
Qed.

(* ======================================================================== *)
(* The source of new names                                                  *)
(* ======================================================================== *)

Theorem no_source_commutes : no_source_commutes_statement.
Proof.
  intros fresh Hrelated.
  specialize
    (Hrelated (Renaming [(fresh [], Pos.succ (fresh []))] []) []).
  cbn [map] in Hrelated.
  unfold rename_variable in Hrelated.
  cbn [variable_permutation permute] in Hrelated. unfold swap in Hrelated.
  rewrite Pos.eqb_refl in Hrelated.
  exact (Pos.succ_discr (fresh []) Hrelated).
Qed.

Theorem rename_fresh_related : rename_fresh_related_statement.
Proof.
  intros r fresh names. unfold rename_fresh.
  rewrite map_rename_variable_inverse_l. reflexivity.
Qed.

Theorem fresh_related_unique : fresh_related_unique_statement.
Proof.
  intros r fresh fresh' Hrelated names. unfold rename_fresh.
  rewrite <- Hrelated, map_rename_variable_inverse_r. reflexivity.
Qed.

Theorem fresh_related_avoids : fresh_related_avoids_statement.
Proof.
  intros r fresh fresh' Hrelated Havoids names Hin.
  rewrite (fresh_related_unique r fresh fresh' Hrelated) in Hin.
  unfold rename_fresh in Hin.
  apply (Havoids (map (rename_variable (inverse_renaming r)) names)).
  apply in_map_iff.
  exists
    (rename_variable r
       (fresh (map (rename_variable (inverse_renaming r)) names))).
  split; [ apply rename_variable_inverse_l | exact Hin ].
Qed.

(* ======================================================================== *)
(* The parameters of the join                                               *)
(* ======================================================================== *)

Theorem join_parameters_related : join_parameters_related_statement.
Proof.
  intros r join_depth. unfold parameters_related.
  cbn [with_fresh join_parameters Join_env.fresh Join_env.type_eqb
       Join_env.env_extension_eqb Join_env.expand_const Join_env.join_depth].
  split; [ apply rename_fresh_related | ].
  split; [ exact (proj1 type_eqb_rename r) | ].
  split; [ exact (proj2 type_eqb_rename r) | ].
  split; [ | reflexivity ].
  intros c. destruct c; try reflexivity.
  match goal with
  | kind : flambda_kind |- _ =>
    destruct kind as [ | k | | ]; try reflexivity; destruct k; reflexivity
  end.
Qed.

Theorem parameters_related_avoids : parameters_related_avoids_statement.
Proof.
  intros r p p' Hrelated Havoids.
  exact
    (fresh_related_avoids r (Join_env.fresh p) (Join_env.fresh p')
       (proj1 Hrelated) Havoids).
Qed.

(* ======================================================================== *)
(* Indices and vectors                                                      *)
(* ======================================================================== *)

(* An index of a joined environment is not a name. A map keyed by
   indices is renamed by what it binds them to. *)

Lemma number_renamed {A B : Type} (g : A -> B) (xs : list A) :
  Index.number (map g xs) = Keyed.map_values g (Index.number xs).
Proof.
  unfold Index.number, Keyed.map_values. rewrite length_map.
  generalize (seq 0 (length xs)) as indices.
  induction xs as [ | x xs IH]; intros [ | index indices];
    cbn [combine map fst snd]; try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma insert_renamed {A B : Type} (g : A -> B) (binding : Index.t * A)
    (sorted : list (Index.t * A)) :
  Index.insert (fst binding, g (snd binding)) (Keyed.map_values g sorted)
  = Keyed.map_values g (Index.insert binding sorted).
Proof.
  unfold Keyed.map_values.
  induction sorted as [ | binding' sorted IH];
    cbn [Index.insert map fst snd]; [ reflexivity | ].
  destruct (Nat.leb (fst binding) (fst binding')); cbn [map fst snd];
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma bindings_renamed {A B : Type} (g : A -> B)
    (map : Keyed.finite_map Index.t A) :
  Index.bindings (Keyed.map_values g map)
  = Keyed.map_values g (Index.bindings map).
Proof.
  unfold Index.bindings. rewrite compact_map_values.
  induction (Keyed.compact Index.equal map) as [ | binding bindings IH];
    [ reflexivity | ].
  cbn [List.fold_right]. rewrite <- insert_renamed, <- IH. reflexivity.
Qed.

Lemma choose_renamed {A B : Type} (g : A -> B)
    (map : Keyed.finite_map Index.t A) :
  Index.choose (Keyed.map_values g map)
  = option_map (fun binding => (fst binding, g (snd binding)))
      (Index.choose map).
Proof.
  unfold Index.choose. rewrite bindings_renamed.
  destruct (Index.bindings map); reflexivity.
Qed.

Lemma simples_find_renamed (r : renaming) (index : Index.t)
    (simples : Simples_in_joined_envs.t) :
  Simples_in_joined_envs.find index (rename_simples_in_joined_envs r simples)
  = option_map (rename_simple r) (Simples_in_joined_envs.find index simples).
Proof. apply keyed_lookup_map_values. Qed.

Lemma simples_equal_renamed (r : renaming)
    (simples simples' : Simples_in_joined_envs.t) :
  Simples_in_joined_envs.equal (rename_simples_in_joined_envs r simples)
    (rename_simples_in_joined_envs r simples')
  = Simples_in_joined_envs.equal simples simples'.
Proof.
  unfold Simples_in_joined_envs.equal, rename_simples_in_joined_envs.
  f_equal.
  - apply for_all_bindings_b_map_values. intros index simple.
    change (Keyed.map_values (rename_simple r) simples')
      with (rename_simples_in_joined_envs r simples').
    rewrite simples_find_renamed.
    destruct (Simples_in_joined_envs.find index simples') as [simple' | ];
      cbn [option_map]; [ apply simple_equal_renamed | reflexivity ].
  - apply for_all_bindings_b_map_values. intros index simple.
    apply is_bound_map_values.
Qed.

(* ======================================================================== *)
(* The source environment                                                   *)
(* ======================================================================== *)

Lemma exists_in_source_env_renamed (r : renaming) (source_env : Source_env.t)
    (var : variable) :
  Source_env.exists_in_source_env (rename_typing_env r source_env)
    (rename_variable r var)
  = Source_env.exists_in_source_env source_env var.
Proof. exact (mem_equivariant r None source_env (Name_var var)). Qed.

Lemma exists_at_name_mode_renamed (r : renaming) (min_name_mode : Name_mode.t)
    (source_env : Source_env.t) (var : variable) :
  Source_env.exists_at_name_mode min_name_mode (rename_typing_env r source_env)
    (rename_variable r var)
  = Source_env.exists_at_name_mode min_name_mode source_env var.
Proof.
  exact (mem_equivariant r (Some min_name_mode) source_env (Name_var var)).
Qed.

Lemma latest_bound_renamed (r : renaming) (source_env : Source_env.t)
    (var var0 : variable) :
  Source_env.latest_bound (rename_typing_env r source_env)
    (rename_variable r var) (rename_variable r var0)
  = option_map (rename_variable r)
      (Source_env.latest_bound source_env var var0).
Proof.
  unfold Source_env.latest_bound. rewrite rename_variable_eqb.
  destruct (Pos.eqb var var0); [ reflexivity | ].
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  change (Name_var (rename_variable r var0))
    with (rename_name r (Name_var var0)).
  rewrite !binding_time_and_mode_equivariant.
  destruct (Typing_env.binding_time_and_mode source_env (Name_var var))
    as [time | ];
    [ | reflexivity ].
  destruct (Typing_env.binding_time_and_mode source_env (Name_var var0))
    as [time0 | ];
    [ | reflexivity ].
  cbv zeta.
  destruct
    (Binding_time.strictly_earlier
       (Binding_time.With_name_mode.binding_time time)
       (Binding_time.With_name_mode.binding_time time0));
    [ reflexivity | ].
  destruct
    (Binding_time.strictly_earlier
       (Binding_time.With_name_mode.binding_time time0)
       (Binding_time.With_name_mode.binding_time time));
    reflexivity.
Qed.

Lemma source_var_of_renamed (r : renaming) (source_env : Source_env.t)
    (simple : simple) :
  Source_env.source_var_of (rename_typing_env r source_env)
    (rename_simple r simple)
  = option_map (rename_variable r)
      (Source_env.source_var_of source_env simple).
Proof.
  destruct simple as [var | sym | c];
    cbn [Source_env.source_var_of rename_simple]; try reflexivity.
  rewrite exists_in_source_env_renamed.
  destruct (Source_env.exists_in_source_env source_env var); reflexivity.
Qed.

Lemma candidate_step_renamed (r : renaming) (source_env : Source_env.t)
    (possible : Source_env.candidate) (canonical : simple) :
  Source_env.candidate_step (rename_typing_env r source_env)
    (rename_candidate r possible) (rename_simple r canonical)
  = option_map (rename_candidate r)
      (Source_env.candidate_step source_env possible canonical).
Proof.
  unfold Source_env.candidate_step. cbv zeta.
  destruct possible as [ | | shared_simple | var];
    cbn [rename_candidate]; rewrite ?source_var_of_renamed.
  - reflexivity.
  - destruct (Source_env.source_var_of source_env canonical); reflexivity.
  - rewrite simple_equal_renamed.
    destruct (simple_equal canonical shared_simple); [ reflexivity | ].
    destruct (Source_env.source_var_of source_env shared_simple) as [var | ];
      cbn [option_map].
    + destruct (Source_env.source_var_of source_env canonical) as [var0 | ];
        cbn [option_map]; [ | reflexivity ].
      rewrite latest_bound_renamed.
      destruct (Source_env.latest_bound source_env var var0); reflexivity.
    + destruct (Source_env.source_var_of source_env canonical); reflexivity.
  - destruct (Source_env.source_var_of source_env canonical) as [var0 | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite latest_bound_renamed.
    destruct (Source_env.latest_bound source_env var var0); reflexivity.
Qed.

Lemma candidate_canonical_in_source_env_renamed (r : renaming)
    (source_env : Source_env.t) (canonicals : Simples_in_joined_envs.t) :
  Source_env.candidate_canonical_in_source_env (rename_typing_env r source_env)
    (rename_simples_in_joined_envs r canonicals)
  = option_map (rename_candidate r)
      (Source_env.candidate_canonical_in_source_env source_env canonicals).
Proof.
  unfold Source_env.candidate_canonical_in_source_env,
    rename_simples_in_joined_envs.
  rewrite compact_map_values.
  change (Some Source_env.No_simples_in_joined_envs)
    with (option_map (rename_candidate r)
            (Some Source_env.No_simples_in_joined_envs))
    at 1.
  apply
    (fold_left_renamed (option_map (rename_candidate r))
       (fun binding : Index.t * simple =>
          (fst binding, rename_simple r (snd binding)))).
  intros [possible | ] binding; cbn [option_map fst snd]; [ | reflexivity ].
  apply candidate_step_renamed.
Qed.

(* ======================================================================== *)
(* The names of the target environment                                      *)
(* ======================================================================== *)

Lemma is_bound_rename_keys {A : Type} (r : renaming) (var : variable)
    (bindings : finite_map A) :
  is_bound (rename_variable r var) (rename_keys r bindings)
  = is_bound var bindings.
Proof.
  exact
    (is_bound_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r)
       (fun a => a) var bindings).
Qed.

Lemma from_source_env_renamed (r : renaming) (names_to_avoid : list variable)
    (source_env : Source_env.t) :
  Bindings_in_target_env.from_source_env
    (map (rename_variable r) names_to_avoid) (rename_typing_env r source_env)
  = rename_bindings_in_target_env r
      (Bindings_in_target_env.from_source_env names_to_avoid source_env).
Proof. reflexivity. Qed.

Lemma exists_in_target_env_renamed (r : renaming)
    (bindings : Bindings_in_target_env.t) (var : variable) :
  Bindings_in_target_env.exists_in_target_env
    (rename_bindings_in_target_env r bindings) (rename_variable r var)
  = Bindings_in_target_env.exists_in_target_env bindings var.
Proof.
  unfold Bindings_in_target_env.exists_in_target_env.
  cbn [rename_bindings_in_target_env Bindings_in_target_env.source_env
       Bindings_in_target_env.existential_variables
       Bindings_in_target_env.imported_variables].
  rewrite exists_in_source_env_renamed, !is_bound_rename_keys. reflexivity.
Qed.

Lemma has_existential_for_these_simples_renamed (r : renaming)
    (bindings : Bindings_in_target_env.t)
    (simples : Simples_in_joined_envs.t) :
  Bindings_in_target_env.has_existential_for_these_simples
    (rename_bindings_in_target_env r bindings)
    (rename_simples_in_joined_envs r simples)
  = option_map (rename_variable r)
      (Bindings_in_target_env.has_existential_for_these_simples bindings
         simples).
Proof.
  exact
    (lookup_renamed Simples_in_joined_envs.equal
       (rename_simples_in_joined_envs r) (simples_equal_renamed r)
       (rename_variable r) simples
       (Bindings_in_target_env.existential_for_these_simples_map bindings)).
Qed.

Theorem existential_for_these_simples_renamed :
  existential_for_these_simples_renamed_statement.
Proof.
  intros r fresh fresh' bindings simples kind Hrelated.
  unfold Bindings_in_target_env.existential_for_these_simples.
  rewrite has_existential_for_these_simples_renamed.
  destruct
    (Bindings_in_target_env.has_existential_for_these_simples bindings
       simples)
    as [existing | ];
    cbn [option_map fst snd]; [ reflexivity | ].
  cbv zeta.
  change
    (Bindings_in_target_env.names_to_avoid
       (rename_bindings_in_target_env r bindings))
    with (map (rename_variable r)
            (Bindings_in_target_env.names_to_avoid bindings)).
  rewrite Hrelated. reflexivity.
Qed.

Lemma import_from_all_envs_renamed (r : renaming)
    (bindings : Bindings_in_target_env.t) (imported_var : variable)
    (kind : flambda_kind) :
  Bindings_in_target_env.import_from_all_envs
    (rename_bindings_in_target_env r bindings)
    (rename_variable r imported_var) kind
  = rename_bindings_in_target_env r
      (Bindings_in_target_env.import_from_all_envs bindings imported_var kind).
Proof.
  unfold Bindings_in_target_env.import_from_all_envs.
  change
    (Bindings_in_target_env.imported_variables
       (rename_bindings_in_target_env r bindings))
    with (rename_keys r (Bindings_in_target_env.imported_variables bindings)).
  rewrite is_bound_rename_keys.
  destruct
    (is_bound imported_var
       (Bindings_in_target_env.imported_variables bindings));
    reflexivity.
Qed.

Lemma compact_rename_keys {A : Type} (r : renaming) (bindings : finite_map A) :
  compact (rename_keys r bindings) = rename_keys r (compact bindings).
Proof.
  exact
    (compact_keys_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r)
       (fun a => a) bindings).
Qed.

Lemma imported_variables_in_order_renamed (r : renaming)
    (bindings : Bindings_in_target_env.t) :
  Bindings_in_target_env.imported_variables_in_order
    (rename_bindings_in_target_env r bindings)
  = rename_keys r
      (Bindings_in_target_env.imported_variables_in_order bindings).
Proof.
  unfold Bindings_in_target_env.imported_variables_in_order.
  change
    (Bindings_in_target_env.imported_variables
       (rename_bindings_in_target_env r bindings))
    with (rename_keys r (Bindings_in_target_env.imported_variables bindings)).
  rewrite compact_rename_keys. unfold rename_keys, Keyed.map_bindings.
  symmetry. apply map_rev.
Qed.

Lemma existential_variables_in_order_renamed (r : renaming)
    (bindings : Bindings_in_target_env.t) :
  Bindings_in_target_env.existential_variables_in_order
    (rename_bindings_in_target_env r bindings)
  = rename_keys r
      (Bindings_in_target_env.existential_variables_in_order bindings).
Proof.
  unfold Bindings_in_target_env.existential_variables_in_order.
  change
    (Bindings_in_target_env.existential_variables
       (rename_bindings_in_target_env r bindings))
    with (rename_keys r
            (Bindings_in_target_env.existential_variables bindings)).
  rewrite compact_rename_keys. unfold rename_keys, Keyed.map_bindings.
  symmetry. apply map_rev.
Qed.

(* ======================================================================== *)
(* The joined environments                                                  *)
(* ======================================================================== *)

Lemma get_nth_joined_env_renamed (r : renaming) (envs : Joined_envs.t)
    (index : Index.t) :
  Joined_envs.get_nth_joined_env (rename_joined_envs r envs) index
  = option_map (rename_typing_env r)
      (Joined_envs.get_nth_joined_env envs index).
Proof.
  unfold Joined_envs.get_nth_joined_env, rename_joined_envs.
  rewrite keyed_lookup_map_values.
  destruct (Keyed.lookup Index.equal index envs) as [[tenv equations] | ];
    reflexivity.
Qed.

Lemma joined_envs_keys_renamed (r : renaming) (envs : Joined_envs.t) :
  Joined_envs.keys (rename_joined_envs r envs) = Joined_envs.keys envs.
Proof. apply domain_map_values. Qed.

Lemma exists_in_all_joined_envs_renamed {A B : Type} (r : renaming)
    (g : A -> B) (envs : Joined_envs.t) (map : Keyed.finite_map Index.t A) :
  Joined_envs.exists_in_all_joined_envs (rename_joined_envs r envs)
    (Keyed.map_values g map)
  = Joined_envs.exists_in_all_joined_envs envs map.
Proof.
  unfold Joined_envs.exists_in_all_joined_envs.
  rewrite joined_envs_keys_renamed.
  induction (Joined_envs.keys envs) as [ | index keys IH]; cbn [forallb];
    [ reflexivity | rewrite is_bound_map_values, IH; reflexivity ].
Qed.

Lemma equal_in_bindings_renamed (r : renaming) (envs : Joined_envs.t)
    (simple : simple) (bindings : list (Index.t * FlambdaModel.simple)) :
  Joined_envs.equal_in_bindings (rename_joined_envs r envs)
    (rename_simple r simple) (Keyed.map_values (rename_simple r) bindings)
  = Joined_envs.equal_in_bindings envs simple bindings.
Proof.
  induction bindings as [ | [index canonical] bindings IH];
    cbn [Joined_envs.equal_in_bindings Keyed.map_values map fst snd];
    [ reflexivity | ].
  rewrite simple_equal_renamed.
  destruct (simple_equal canonical simple); [ exact IH | ].
  rewrite get_nth_joined_env_renamed.
  destruct (Joined_envs.get_nth_joined_env envs index) as [env | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite get_canonical_ignoring_name_mode_equivariant, simple_equal_renamed.
  destruct
    (simple_equal canonical
       (Typing_env.get_canonical_simple_ignoring_name_mode env simple));
    [ exact IH | reflexivity ].
Qed.

Lemma equal_in_all_joined_envs_renamed (r : renaming) (envs : Joined_envs.t)
    (simple : simple) (simples : Simples_in_joined_envs.t) :
  Joined_envs.equal_in_all_joined_envs (rename_joined_envs r envs)
    (rename_simple r simple) (rename_simples_in_joined_envs r simples)
  = Joined_envs.equal_in_all_joined_envs envs simple simples.
Proof.
  unfold Joined_envs.equal_in_all_joined_envs, rename_simples_in_joined_envs.
  rewrite compact_map_values. apply equal_in_bindings_renamed.
Qed.

Lemma canonicals_of_alias_types_renamed (r : renaming) (envs : Joined_envs.t)
    (types : list (Index.t * flambda_type)) :
  Joined_envs.canonicals_of_alias_types (rename_joined_envs r envs)
    (Keyed.map_values (rename_type r) types)
  = option_map (option_map (rename_simples_in_joined_envs r))
      (Joined_envs.canonicals_of_alias_types envs types).
Proof.
  induction types as [ | [index ty] types IH];
    cbn [Joined_envs.canonicals_of_alias_types Keyed.map_values map fst snd];
    [ reflexivity | ].
  rewrite get_nth_joined_env_renamed.
  destruct (Joined_envs.get_nth_joined_env envs index) as [env | ];
    cbn [option_map]; [ | reflexivity ].
  change (map (fun binding => (fst binding, rename_type r (snd binding))) types)
    with (Keyed.map_values (rename_type r) types).
  rewrite IH.
  destruct (Joined_envs.canonicals_of_alias_types envs types)
    as [canonicals | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite (proj2 (rename_type_alias r)).
  destruct (get_alias_opt ty) as [alias | ], canonicals as [canonicals | ];
    cbn [option_map]; try reflexivity.
  rewrite get_canonical_ignoring_name_mode_equivariant. reflexivity.
Qed.

Lemma get_alias_then_canonical_simple_ignoring_name_mode_exn_renamed
    (r : renaming) (envs : Joined_envs.t)
    (types : Keyed.finite_map Index.t flambda_type) :
  Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
    (rename_joined_envs r envs) (Keyed.map_values (rename_type r) types)
  = option_map (option_map (rename_simples_in_joined_envs r))
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         envs types).
Proof.
  unfold Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn.
  rewrite compact_map_values. apply canonicals_of_alias_types_renamed.
Qed.

(* The type of a constant is given by a function: on the renamed side
   it is to give the renamed type. *)
Lemma types_of_simples_renamed (r : renaming)
    (expand_const expand_const' : const -> flambda_type)
    (envs : Joined_envs.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) :
  (∀ c : const, expand_const' c = rename_type r (expand_const c)) ->
  Joined_envs.types_of_simples expand_const' (rename_joined_envs r envs) kind
    (Keyed.map_values (rename_simple r) simples)
  = option_map (Keyed.map_values (rename_type r))
      (Joined_envs.types_of_simples expand_const envs kind simples).
Proof.
  intros Hconst.
  induction simples as [ | [index simple] simples IH];
    cbn [Joined_envs.types_of_simples Keyed.map_values map fst snd];
    [ reflexivity | ].
  change
    (map (fun binding => (fst binding, rename_simple r (snd binding))) simples)
    with (Keyed.map_values (rename_simple r) simples).
  rewrite IH.
  destruct simple as [var | sym | c]; cbn [rename_simple].
  - rewrite get_nth_joined_env_renamed.
    destruct (Joined_envs.get_nth_joined_env envs index) as [env | ];
      cbn [option_map]; [ | reflexivity ].
    change (Name_var (rename_variable r var))
      with (rename_name r (Name_var var)).
    rewrite find_equivariant.
    destruct (Typing_env.find env (Name_var var) (Some kind)) as [ty | ];
      cbn [option_map]; [ | reflexivity ].
    destruct (Joined_envs.types_of_simples expand_const envs kind simples);
      reflexivity.
  - rewrite get_nth_joined_env_renamed.
    destruct (Joined_envs.get_nth_joined_env envs index) as [env | ];
      cbn [option_map]; [ | reflexivity ].
    change (Name_symbol sym) with (rename_name r (Name_symbol sym)) at 1.
    rewrite find_equivariant.
    destruct (Typing_env.find env (Name_symbol sym) (Some kind)) as [ty | ];
      cbn [option_map]; [ | reflexivity ].
    destruct (Joined_envs.types_of_simples expand_const envs kind simples);
      reflexivity.
  - rewrite Hconst.
    destruct (Joined_envs.types_of_simples expand_const envs kind simples);
      reflexivity.
Qed.

Lemma find_simples_in_joined_envs_renamed (r : renaming)
    (expand_const expand_const' : const -> flambda_type)
    (envs : Joined_envs.t) (simples : Simples_in_joined_envs.t)
    (kind : flambda_kind) :
  (∀ c : const, expand_const' c = rename_type r (expand_const c)) ->
  Joined_envs.find_simples_in_joined_envs expand_const'
    (rename_joined_envs r envs) (rename_simples_in_joined_envs r simples) kind
  = option_map (Keyed.map_values (rename_type r))
      (Joined_envs.find_simples_in_joined_envs expand_const envs simples
         kind).
Proof.
  intros Hconst.
  unfold Joined_envs.find_simples_in_joined_envs,
    rename_simples_in_joined_envs.
  rewrite compact_map_values. apply types_of_simples_renamed, Hconst.
Qed.

(* A joined environment with the equations of its level, renamed. *)
Local Abbreviation rename_joined r :=
  (fun joined : Typing_env.t * Keyed.finite_map name flambda_type =>
     (rename_typing_env r (fst joined),
      Keyed.map_bindings (rename_name r) (rename_type r) (snd joined))).

Lemma types_of_imported_var_renamed (r : renaming)
    (envs :
       list (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
    (var : variable) (kind : flambda_kind) :
  Joined_envs.types_of_imported_var
    (Keyed.map_values (rename_joined r) envs) (rename_variable r var) kind
  = option_map (Keyed.map_values (rename_type r))
      (Joined_envs.types_of_imported_var envs var kind).
Proof.
  induction envs as [ | [index [env equations]] envs IH];
    cbn [Joined_envs.types_of_imported_var Keyed.map_values map fst snd];
    [ reflexivity | ].
  change
    (map
       (fun binding
            : Index.t
              * (Typing_env.t * Keyed.finite_map name flambda_type) =>
          (fst binding,
           (rename_typing_env r (fst (snd binding)),
            Keyed.map_bindings (rename_name r) (rename_type r)
              (snd (snd binding)))))
       envs)
    with (Keyed.map_values (rename_joined r) envs).
  rewrite IH.
  destruct (Joined_envs.types_of_imported_var envs var kind) as [types | ];
    cbn [option_map]; [ | reflexivity ].
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  rewrite mem_equivariant.
  destruct (Typing_env.mem None env (Name_var var)); [ | reflexivity ].
  rewrite find_equivariant.
  destruct (Typing_env.find env (Name_var var) (Some kind)); reflexivity.
Qed.

Lemma find_imported_var_renamed (r : renaming) (envs : Joined_envs.t)
    (var : variable) (kind : flambda_kind) :
  Joined_envs.find_imported_var (rename_joined_envs r envs)
    (rename_variable r var) kind
  = option_map (Keyed.map_values (rename_type r))
      (Joined_envs.find_imported_var envs var kind).
Proof.
  unfold Joined_envs.find_imported_var, rename_joined_envs.
  rewrite compact_map_values. apply types_of_imported_var_renamed.
Qed.

Lemma kind_of_var_in_renamed (r : renaming)
    (envs :
       list (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
    (var : variable) :
  Joined_envs.kind_of_var_in (Keyed.map_values (rename_joined r) envs)
    (rename_variable r var)
  = Joined_envs.kind_of_var_in envs var.
Proof.
  induction envs as [ | [index [env equations]] envs IH];
    cbn [Joined_envs.kind_of_var_in Keyed.map_values map fst snd];
    [ reflexivity | ].
  change
    (map
       (fun binding
            : Index.t
              * (Typing_env.t * Keyed.finite_map name flambda_type) =>
          (fst binding,
           (rename_typing_env r (fst (snd binding)),
            Keyed.map_bindings (rename_name r) (rename_type r)
              (snd (snd binding)))))
       envs)
    with (Keyed.map_values (rename_joined r) envs).
  rewrite IH.
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  rewrite find_equivariant, mem_equivariant.
  destruct (Typing_env.find env (Name_var var) None) as [ty | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite kind_rename_type. reflexivity.
Qed.

Lemma kind_of_imported_var_renamed (r : renaming) (envs : Joined_envs.t)
    (var : variable) :
  Joined_envs.kind_of_imported_var (rename_joined_envs r envs)
    (rename_variable r var)
  = Joined_envs.kind_of_imported_var envs var.
Proof.
  unfold Joined_envs.kind_of_imported_var, rename_joined_envs.
  rewrite compact_map_values. apply kind_of_var_in_renamed.
Qed.

Lemma get_canonical_simples_ignoring_name_mode_renamed (r : renaming)
    (envs : Joined_envs.t) (simples : list (Index.t * simple)) :
  Joined_envs.get_canonical_simples_ignoring_name_mode
    (rename_joined_envs r envs) (Keyed.map_values (rename_simple r) simples)
  = option_map (rename_simples_in_joined_envs r)
      (Joined_envs.get_canonical_simples_ignoring_name_mode envs simples).
Proof.
  unfold Joined_envs.get_canonical_simples_ignoring_name_mode.
  match goal with
  | |- List.fold_left _ _ ?init = _ =>
    change init with (option_map (rename_simples_in_joined_envs r) init)
      at 1
  end.
  apply
    (fold_left_renamed (option_map (rename_simples_in_joined_envs r))
       (fun binding : Index.t * simple =>
          (fst binding, rename_simple r (snd binding)))).
  intros [acc | ] [index simple]; cbn [option_map fst snd];
    [ | reflexivity ].
  rewrite get_nth_joined_env_renamed.
  destruct (Joined_envs.get_nth_joined_env envs index) as [env | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite get_canonical_ignoring_name_mode_equivariant. reflexivity.
Qed.

(* ======================================================================== *)
(* Maps keyed by variables                                                  *)
(* ======================================================================== *)

Lemma lookup_renamed_keys {A B : Type} (r : renaming) (g : A -> B)
    (var : variable) (bindings : finite_map A) :
  lookup (rename_variable r var)
    (Keyed.map_bindings (rename_variable r) g bindings)
  = option_map g (lookup var bindings).
Proof.
  exact
    (lookup_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r) g var
       bindings).
Qed.

Lemma is_bound_renamed_keys {A B : Type} (r : renaming) (g : A -> B)
    (var : variable) (bindings : finite_map A) :
  is_bound (rename_variable r var)
    (Keyed.map_bindings (rename_variable r) g bindings)
  = is_bound var bindings.
Proof.
  exact
    (is_bound_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r) g
       var bindings).
Qed.

Lemma among_renamed_variables (r : renaming) (var : variable)
    (names : list variable) :
  among (rename_variable r var) (map (rename_variable r) names)
  = among var names.
Proof.
  exact
    (among_keys_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r)
       var names).
Qed.

Lemma filter_bindings_renamed_keys {A B : Type} (r : renaming) (g : A -> B)
    (keep : variable -> A -> bool) (keep' : variable -> B -> bool)
    (bindings : finite_map A) :
  (∀ (var : variable) (a : A), keep' (rename_variable r var) (g a) = keep var a)
  ->
  filter_bindings keep' (Keyed.map_bindings (rename_variable r) g bindings)
  = Keyed.map_bindings (rename_variable r) g (filter_bindings keep bindings).
Proof.
  exact
    (filter_bindings_renamed Pos.eqb (rename_variable r)
       (rename_variable_eqb r) g keep keep' bindings).
Qed.

(* ======================================================================== *)
(* The aliases of the variables made                                        *)
(* ======================================================================== *)

Lemma aliases_of_existential_var_renamed (r : renaming)
    (aliases : Aliases_of_existentials.t) (var : variable) :
  Aliases_of_existentials.aliases_of_existential_var
    (rename_aliases_of_existentials r aliases) (rename_variable r var)
  = map (rename_variable r)
      (Aliases_of_existentials.aliases_of_existential_var aliases var).
Proof.
  unfold Aliases_of_existentials.aliases_of_existential_var,
    rename_aliases_of_existentials.
  rewrite lookup_renamed_keys.
  destruct (lookup var aliases); reflexivity.
Qed.

Lemma aliases_add_renamed (r : renaming) (aliases : Aliases_of_existentials.t)
    (demoted_var existential_var : variable) :
  Aliases_of_existentials.add (rename_aliases_of_existentials r aliases)
    (rename_variable r demoted_var) (rename_variable r existential_var)
  = rename_aliases_of_existentials r
      (Aliases_of_existentials.add aliases demoted_var existential_var).
Proof.
  unfold Aliases_of_existentials.add. cbv zeta.
  rewrite aliases_of_existential_var_renamed, among_renamed_variables.
  destruct
    (among demoted_var
       (Aliases_of_existentials.aliases_of_existential_var aliases
          existential_var));
    reflexivity.
Qed.

(* ======================================================================== *)
(* The state of a join                                                      *)
(* ======================================================================== *)

Lemma join_env_create_renamed (r : renaming) (joined_envs : Joined_envs.t)
    (bindings : Bindings_in_target_env.t) :
  Join_env.create (rename_joined_envs r joined_envs)
    (rename_bindings_in_target_env r bindings)
  = rename_join_env r (Join_env.create joined_envs bindings).
Proof. reflexivity. Qed.

Lemma join_env_source_env_renamed (r : renaming) (env : Join_env.t) :
  Join_env.source_env (rename_join_env r env)
  = rename_typing_env r (Join_env.source_env env).
Proof. reflexivity. Qed.

Lemma new_definitions_of_existentials_renamed (r : renaming)
    (env since : Join_env.t) :
  Join_env.new_definitions_of_existentials (rename_join_env r env)
    (rename_join_env r since)
  = Keyed.map_bindings (rename_variable r)
      (fun definition : Simples_in_joined_envs.t * flambda_kind =>
         (rename_simples_in_joined_envs r (fst definition), snd definition))
      (Join_env.new_definitions_of_existentials env since).
Proof.
  unfold Join_env.new_definitions_of_existentials.
  cbn [rename_join_env Join_env.definitions_of_existentials].
  apply filter_bindings_renamed_keys. intros var definition.
  rewrite is_bound_renamed_keys. reflexivity.
Qed.

Lemma new_equations_in_joined_envs_renamed (r : renaming)
    (env since : Join_env.t) :
  Join_env.new_equations_in_joined_envs (rename_join_env r env)
    (rename_join_env r since)
  = Keyed.map_bindings (rename_variable r) (Keyed.map_values (rename_type r))
      (Join_env.new_equations_in_joined_envs env since).
Proof.
  unfold Join_env.new_equations_in_joined_envs.
  cbn [rename_join_env Join_env.types_in_joined_envs].
  apply filter_bindings_renamed_keys. intros var types.
  rewrite is_bound_renamed_keys. reflexivity.
Qed.

Lemma with_bindings_renamed (r : renaming) (env : Join_env.t)
    (bindings : Bindings_in_target_env.t) :
  Join_env.with_bindings (rename_join_env r env)
    (rename_bindings_in_target_env r bindings)
  = rename_join_env r (Join_env.with_bindings env bindings).
Proof. reflexivity. Qed.

Theorem existential_for_these_canonicals_renamed :
  existential_for_these_canonicals_renamed_statement.
Proof.
  intros r p p' env canonicals kind Hrelated.
  unfold Join_env.existential_for_these_canonicals.
  change (Join_env.bindings (rename_join_env r env))
    with (rename_bindings_in_target_env r (Join_env.bindings env)).
  rewrite
    (existential_for_these_simples_renamed r (Join_env.fresh p)
       (Join_env.fresh p') (Join_env.bindings env) canonicals kind
       (proj1 Hrelated)).
  destruct
    (Bindings_in_target_env.existential_for_these_simples (Join_env.fresh p)
       (Join_env.bindings env) canonicals kind)
    as [existential_var bindings].
  cbn [fst snd rename_join_env Join_env.definitions_of_existentials].
  rewrite is_bound_renamed_keys.
  destruct
    (is_bound existential_var (Join_env.definitions_of_existentials env));
    reflexivity.
Qed.

Lemma add_alias_of_existential_renamed (r : renaming)
    (p p' : Join_env.parameters) (env : Join_env.t) (kind : flambda_kind)
    (var : variable) (canonicals : Simples_in_joined_envs.t) :
  parameters_related r p p' ->
  Join_env.add_alias_of_existential p' (rename_join_env r env) kind
    (rename_variable r var) (rename_simples_in_joined_envs r canonicals)
  = rename_join_env r
      (Join_env.add_alias_of_existential p env kind var canonicals).
Proof.
  intros Hrelated. unfold Join_env.add_alias_of_existential.
  rewrite
    (existential_for_these_canonicals_renamed r p p' env canonicals kind
       Hrelated).
  destruct (Join_env.existential_for_these_canonicals p env canonicals kind)
    as [existential_var env'].
  cbn [fst snd]. unfold rename_join_env at 2.
  cbn [rename_join_env Join_env.joined_envs Join_env.types_in_target_env
       Join_env.types_in_joined_envs Join_env.aliases_of_existentials
       Join_env.imported_vars Join_env.definitions_of_existentials
       Join_env.bindings].
  rewrite aliases_add_renamed. reflexivity.
Qed.

Lemma add_equation_in_target_env_renamed (r : renaming) (env : Join_env.t)
    (var : variable) (ty : flambda_type) :
  Join_env.add_equation_in_target_env (rename_join_env r env)
    (rename_variable r var) (rename_type r ty)
  = option_map (rename_join_env r)
      (Join_env.add_equation_in_target_env env var ty).
Proof.
  unfold Join_env.add_equation_in_target_env.
  cbn [rename_join_env Join_env.types_in_target_env].
  rewrite is_bound_renamed_keys.
  destruct (is_bound var (Join_env.types_in_target_env env)); reflexivity.
Qed.

Lemma add_equations_in_joined_envs_renamed (r : renaming) (env : Join_env.t)
    (var : variable) (types : Keyed.finite_map Index.t flambda_type) :
  Join_env.add_equations_in_joined_envs (rename_join_env r env)
    (rename_variable r var) (Keyed.map_values (rename_type r) types)
  = rename_join_env r (Join_env.add_equations_in_joined_envs env var types).
Proof. reflexivity. Qed.

(* What stands in the target for a vector of canonical [Simple]s,
   renamed. *)
Definition rename_canonical_in_target_env (r : renaming)
    (canonical : Join_env.canonical_in_target_env)
    : Join_env.canonical_in_target_env :=
  match canonical with
  | Join_env.Canonical_in_source_env simple =>
    Join_env.Canonical_in_source_env (rename_simple r simple)
  | Join_env.Import_from_all_joined_envs var =>
    Join_env.Import_from_all_joined_envs (rename_variable r var)
  | Join_env.Existential_for_these_canonicals canonicals =>
    Join_env.Existential_for_these_canonicals
      (rename_simples_in_joined_envs r canonicals)
  end.

Lemma n_way_join_canonical_simples_renamed (r : renaming) (env : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) :
  Join_env.n_way_join_canonical_simples (rename_join_env r env)
    (rename_simples_in_joined_envs r canonicals)
  = option_map (rename_canonical_in_target_env r)
      (Join_env.n_way_join_canonical_simples env canonicals).
Proof.
  unfold Join_env.n_way_join_canonical_simples.
  rewrite join_env_source_env_renamed,
    candidate_canonical_in_source_env_renamed.
  destruct
    (Source_env.candidate_canonical_in_source_env (Join_env.source_env env)
       canonicals)
    as [[ | | simple | var] | ];
    cbn [option_map rename_candidate]; try reflexivity.
  - destruct simple as [var | sym | c]; cbn [rename_simple];
      try reflexivity.
    rewrite exists_in_source_env_renamed.
    destruct (Source_env.exists_in_source_env (Join_env.source_env env) var);
      reflexivity.
  - change (Var (rename_variable r var)) with (rename_simple r (Var var)).
    change (Join_env.joined_envs (rename_join_env r env))
      with (rename_joined_envs r (Join_env.joined_envs env)).
    rewrite equal_in_all_joined_envs_renamed.
    destruct
      (Joined_envs.equal_in_all_joined_envs (Join_env.joined_envs env)
         (Var var) canonicals)
      as [[ | ] | ];
      reflexivity.
Qed.

(* ======================================================================== *)
(* Importing a variable                                                     *)
(* ======================================================================== *)

(* The functions that import a variable are one recursion; each takes
   the function that imports a variable. Two such functions are
   related by a renaming: on the renamed state and the renamed
   variable, the second answers the renamed state of the first. *)
Definition import_var_related (r : renaming)
    (import_var import_var' : Join_env.t -> variable -> option Join_env.t)
    : Prop :=
  ∀ (env : Join_env.t) (var : variable),
    import_var' (rename_join_env r env) (rename_variable r var)
    = option_map (rename_join_env r) (import_var env var).

Lemma import_type_transitive_with_renamed (r : renaming)
    (import_var import_var' : Join_env.t -> variable -> option Join_env.t)
    (env : Join_env.t) (ty : flambda_type) :
  import_var_related r import_var import_var' ->
  Join_env.import_type_transitive_with import_var' (rename_join_env r env)
    (rename_type r ty)
  = option_map (rename_join_env r)
      (Join_env.import_type_transitive_with import_var env ty).
Proof.
  intros Himport. unfold Join_env.import_type_transitive_with.
  rewrite (proj1 free_variables_rename_type).
  change (Some (rename_join_env r env))
    with (option_map (rename_join_env r) (Some env)).
  apply (fold_left_renamed (option_map (rename_join_env r))
           (rename_variable r)).
  intros [env' | ] var; cbn [option_map]; [ apply Himport | reflexivity ].
Qed.

Lemma add_equals_in_joined_envs_with_renamed (r : renaming)
    (p p' : Join_env.parameters)
    (import_var import_var' : Join_env.t -> variable -> option Join_env.t)
    (env : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  parameters_related r p p' ->
  import_var_related r import_var import_var' ->
  Join_env.add_equals_in_joined_envs_with p' import_var'
    (rename_join_env r env) kind (rename_variable r var)
    (rename_simples_in_joined_envs r simples)
  = option_map (rename_join_env r)
      (Join_env.add_equals_in_joined_envs_with p import_var env kind var
         simples).
Proof.
  intros Hp Himport. unfold Join_env.add_equals_in_joined_envs_with.
  rewrite n_way_join_canonical_simples_renamed.
  destruct (Join_env.n_way_join_canonical_simples env simples)
    as [[canonical | imported | canonicals] | ];
    cbn [option_map rename_canonical_in_target_env]; [ | | | reflexivity ].
  - rewrite <- (proj1 (rename_type_alias r)).
    apply add_equation_in_target_env_renamed.
  - rewrite Himport.
    destruct (import_var env imported) as [env' | ]; cbn [option_map];
      [ | reflexivity ].
    change (Var (rename_variable r imported))
      with (rename_simple r (Var imported)).
    rewrite <- (proj1 (rename_type_alias r)).
    apply add_equation_in_target_env_renamed.
  - f_equal. apply add_alias_of_existential_renamed, Hp.
Qed.

Lemma import_or_delay_n_way_join_of_concrete_type_with_renamed (r : renaming)
    (p p' : Join_env.parameters)
    (import_var import_var' : Join_env.t -> variable -> option Join_env.t)
    (env : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  parameters_related r p p' ->
  import_var_related r import_var import_var' ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p' import_var'
    (rename_join_env r env) (rename_variable r var)
    (Keyed.map_values (rename_type r) types)
  = option_map (rename_join_env r)
      (Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var
         env var types).
Proof.
  intros Hp Himport.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with.
  rewrite choose_renamed.
  destruct (Index.choose types) as [[index any_ty] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite compact_map_values.
  assert (Hall :
    forallb
      (fun binding : Index.t * flambda_type =>
         Index.equal (fst binding) index
         || Join_env.type_eqb p' (rename_type r any_ty) (snd binding))
      (Keyed.map_values (rename_type r) (Keyed.compact Index.equal types))
    = forallb
        (fun binding : Index.t * flambda_type =>
           Index.equal (fst binding) index
           || Join_env.type_eqb p any_ty (snd binding))
        (Keyed.compact Index.equal types)).
  { unfold Keyed.map_values. apply forallb_renamed. intros [index' ty].
    cbn [fst snd]. rewrite (proj1 (proj2 Hp)). reflexivity. }
  rewrite Hall.
  destruct
    (forallb
       (fun binding : Index.t * flambda_type =>
          Index.equal (fst binding) index
          || Join_env.type_eqb p any_ty (snd binding))
       (Keyed.compact Index.equal types));
    [ | reflexivity ].
  rewrite add_equation_in_target_env_renamed.
  destruct (Join_env.add_equation_in_target_env env var any_ty)
    as [env' | ];
    cbn [option_map]; [ | reflexivity ].
  apply import_type_transitive_with_renamed, Himport.
Qed.

Lemma import_or_delay_n_way_join_type_with_renamed (r : renaming)
    (p p' : Join_env.parameters)
    (import_var import_var' : Join_env.t -> variable -> option Join_env.t)
    (env : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  parameters_related r p p' ->
  import_var_related r import_var import_var' ->
  Join_env.import_or_delay_n_way_join_type_with p' import_var'
    (rename_join_env r env) (rename_variable r var)
    (Keyed.map_values (rename_type r) types)
  = option_map (rename_join_env r)
      (Join_env.import_or_delay_n_way_join_type_with p import_var env var
         types).
Proof.
  intros Hp Himport. unfold Join_env.import_or_delay_n_way_join_type_with.
  rewrite choose_renamed.
  destruct (Index.choose types) as [[index any_ty] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  change (Join_env.joined_envs (rename_join_env r env))
    with (rename_joined_envs r (Join_env.joined_envs env)).
  rewrite get_alias_then_canonical_simple_ignoring_name_mode_exn_renamed.
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
       (Join_env.joined_envs env) types)
    as [[canonical_simples | ] | ];
    cbn [option_map]; [ | | reflexivity ].
  - rewrite kind_rename_type.
    apply add_equals_in_joined_envs_with_renamed; assumption.
  - apply import_or_delay_n_way_join_of_concrete_type_with_renamed;
      assumption.
Qed.

Lemma existsb_renamed {B C : Type} (g : B -> C) (p : B -> bool)
    (p' : C -> bool) (l : list B) :
  (∀ b : B, p' (g b) = p b) -> existsb p' (map g l) = existsb p l.
Proof.
  intros Hp. induction l as [ | b l IH]; cbn [map existsb]; [ reflexivity | ].
  rewrite Hp, IH. reflexivity.
Qed.

(* The state in which a variable is marked as imported: what
   [import_var_transitive_body] goes on with. *)
Definition mark_imported (env : Join_env.t) (var : variable)
    (kind : flambda_kind) : Join_env.t :=
  {| Join_env.joined_envs := Join_env.joined_envs env;
     Join_env.types_in_target_env := Join_env.types_in_target_env env;
     Join_env.types_in_joined_envs := Join_env.types_in_joined_envs env;
     Join_env.aliases_of_existentials := Join_env.aliases_of_existentials env;
     Join_env.imported_vars := var :: Join_env.imported_vars env;
     Join_env.definitions_of_existentials :=
       Join_env.definitions_of_existentials env;
     Join_env.bindings :=
       Bindings_in_target_env.import_from_all_envs (Join_env.bindings env) var
         kind |}.

Lemma mark_imported_renamed (r : renaming) (env : Join_env.t) (var : variable)
    (kind : flambda_kind) :
  mark_imported (rename_join_env r env) (rename_variable r var) kind
  = rename_join_env r (mark_imported env var kind).
Proof.
  unfold mark_imported, rename_join_env.
  cbn [Join_env.joined_envs Join_env.types_in_target_env
       Join_env.types_in_joined_envs Join_env.aliases_of_existentials
       Join_env.imported_vars Join_env.definitions_of_existentials
       Join_env.bindings map].
  rewrite import_from_all_envs_renamed. reflexivity.
Qed.

Lemma import_var_transitive_body_renamed (r : renaming)
    (p p' : Join_env.parameters)
    (import_var import_var' : Join_env.t -> variable -> option Join_env.t)
    (env : Join_env.t) (var : variable) :
  parameters_related r p p' ->
  import_var_related r import_var import_var' ->
  Join_env.import_var_transitive_body p' import_var' (rename_join_env r env)
    (rename_variable r var)
  = option_map (rename_join_env r)
      (Join_env.import_var_transitive_body p import_var env var).
Proof.
  intros Hp Himport. unfold Join_env.import_var_transitive_body.
  rewrite join_env_source_env_renamed, exists_in_source_env_renamed.
  destruct (Source_env.exists_in_source_env (Join_env.source_env env) var);
    [ reflexivity | ].
  change
    (among (rename_variable r var)
       (Join_env.imported_vars (rename_join_env r env)))
    with (among (rename_variable r var)
            (map (rename_variable r) (Join_env.imported_vars env))).
  rewrite among_renamed_variables.
  destruct (among var (Join_env.imported_vars env)); [ reflexivity | ].
  change
    (Joined_envs.kind_of_imported_var
       (Join_env.joined_envs (rename_join_env r env)) (rename_variable r var))
    with (Joined_envs.kind_of_imported_var
            (rename_joined_envs r (Join_env.joined_envs env))
            (rename_variable r var)).
  rewrite kind_of_imported_var_renamed.
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs env) var)
    as [kind | ];
    [ | reflexivity ].
  fold (mark_imported (rename_join_env r env) (rename_variable r var) kind).
  fold (mark_imported env var kind).
  rewrite mark_imported_renamed. cbv zeta.
  generalize (mark_imported env var kind) as imported. intros imported.
  change (Join_env.joined_envs (rename_join_env r imported))
    with (rename_joined_envs r (Join_env.joined_envs imported)).
  rewrite find_imported_var_renamed.
  destruct
    (Joined_envs.find_imported_var (Join_env.joined_envs imported) var kind)
    as [types | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite compact_map_values.
  assert (Hexists :
    existsb
      (fun binding : Index.t * flambda_type =>
         is_some (get_alias_opt (snd binding)))
      (Keyed.map_values (rename_type r) (Keyed.compact Index.equal types))
    = existsb
        (fun binding : Index.t * flambda_type =>
           is_some (get_alias_opt (snd binding)))
        (Keyed.compact Index.equal types)).
  { unfold Keyed.map_values. apply existsb_renamed. intros [index ty].
    cbn [fst snd]. rewrite (proj2 (rename_type_alias r)).
    destruct (get_alias_opt ty); reflexivity. }
  rewrite Hexists.
  destruct
    (existsb
       (fun binding : Index.t * flambda_type =>
          is_some (get_alias_opt (snd binding)))
       (Keyed.compact Index.equal types));
    [ | apply import_or_delay_n_way_join_of_concrete_type_with_renamed;
        assumption ].
  assert (Htypes :
    Keyed.map_values
      (fun ty : flambda_type =>
         match get_alias_opt ty with
         | None => alias_type_of kind (Var (rename_variable r var))
         | Some _ => ty
         end)
      (Keyed.map_values (rename_type r) types)
    = Keyed.map_values (rename_type r)
        (Keyed.map_values
           (fun ty : flambda_type =>
              match get_alias_opt ty with
              | None => alias_type_of kind (Var var)
              | Some _ => ty
              end)
           types)).
  { unfold Keyed.map_values. rewrite !map_map. apply map_ext.
    intros [index ty]. cbn [fst snd]. f_equal.
    rewrite (proj2 (rename_type_alias r)).
    destruct (get_alias_opt ty); cbn [option_map]; [ reflexivity | ].
    symmetry. exact (proj1 (rename_type_alias r) kind (Var var)). }
  rewrite Htypes,
    get_alias_then_canonical_simple_ignoring_name_mode_exn_renamed.
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
       (Join_env.joined_envs imported)
       (Keyed.map_values
          (fun ty : flambda_type =>
             match get_alias_opt ty with
             | None => alias_type_of kind (Var var)
             | Some _ => ty
             end)
          types))
    as [[canonical_simples | ] | ];
    cbn [option_map]; try reflexivity.
  apply add_equals_in_joined_envs_with_renamed; assumption.
Qed.

Lemma import_var_transitive_renamed (r : renaming)
    (p p' : Join_env.parameters) (fuel : nat) :
  parameters_related r p p' ->
  import_var_related r (Join_env.import_var_transitive p fuel)
    (Join_env.import_var_transitive p' fuel).
Proof.
  intros Hp. induction fuel as [ | fuel IH]; intros env var;
    cbn [Join_env.import_var_transitive]; [ reflexivity | ].
  apply import_var_transitive_body_renamed; assumption.
Qed.

Lemma import_type_transitive_renamed (r : renaming)
    (p p' : Join_env.parameters) (fuel : nat) (env : Join_env.t)
    (ty : flambda_type) :
  parameters_related r p p' ->
  Join_env.import_type_transitive p' fuel (rename_join_env r env)
    (rename_type r ty)
  = option_map (rename_join_env r)
      (Join_env.import_type_transitive p fuel env ty).
Proof.
  intros Hp.
  apply import_type_transitive_with_renamed, import_var_transitive_renamed,
    Hp.
Qed.

Lemma import_or_delay_n_way_join_type_renamed (r : renaming)
    (p p' : Join_env.parameters) (fuel : nat) (env : Join_env.t)
    (var : variable) (types : Keyed.finite_map Index.t flambda_type) :
  parameters_related r p p' ->
  Join_env.import_or_delay_n_way_join_type p' fuel (rename_join_env r env)
    (rename_variable r var) (Keyed.map_values (rename_type r) types)
  = option_map (rename_join_env r)
      (Join_env.import_or_delay_n_way_join_type p fuel env var types).
Proof.
  intros Hp.
  apply import_or_delay_n_way_join_type_with_renamed;
    [ exact Hp | apply import_var_transitive_renamed, Hp ].
Qed.

Lemma import_name_transitive_renamed (r : renaming)
    (p p' : Join_env.parameters) (fuel : nat) (env : Join_env.t)
    (name : name) :
  parameters_related r p p' ->
  Join_env.import_name_transitive p' fuel (rename_join_env r env)
    (rename_name r name)
  = option_map (rename_join_env r)
      (Join_env.import_name_transitive p fuel env name).
Proof.
  intros Hp. destruct name as [var | sym];
    cbn [Join_env.import_name_transitive rename_name]; [ | reflexivity ].
  apply import_var_transitive_renamed, Hp.
Qed.

(* ======================================================================== *)
(* The choices that the join is handed                                      *)
(* ======================================================================== *)

Lemma stands_for_renamed (r : renaming) (env : Join_env.t) (index : Index.t)
    (var : variable) :
  Join_env.stands_for (rename_join_env r env) index (rename_variable r var)
  = option_map (rename_simple r) (Join_env.stands_for env index var).
Proof.
  unfold Join_env.stands_for.
  cbn [rename_join_env Join_env.definitions_of_existentials].
  rewrite lookup_renamed_keys.
  destruct (lookup var (Join_env.definitions_of_existentials env))
    as [[simples kind] | ];
    cbn [option_map fst snd]; [ apply simples_find_renamed | reflexivity ].
Qed.

Lemma has_alias_in_terms_renamed (r : renaming) (tenv : Typing_env.t)
    (simple : simple) :
  Join_env.has_alias_in_terms (rename_typing_env r tenv)
    (rename_simple r simple)
  = Join_env.has_alias_in_terms tenv simple.
Proof.
  unfold Join_env.has_alias_in_terms.
  rewrite get_canonical_simple_exn_equivariant.
  destruct
    (Typing_env.get_canonical_simple_exn tenv (Some Name_mode.normal) None
       simple)
    as [[canonical | ] | ];
    cbn [option_map]; try reflexivity.
  apply mem_simple_equivariant.
Qed.

Lemma in_terms_in_all_joined_envs_renamed (r : renaming) (env : Join_env.t)
    (var : variable) :
  Join_env.in_terms_in_all_joined_envs (rename_join_env r env)
    (rename_variable r var)
  = Join_env.in_terms_in_all_joined_envs env var.
Proof.
  unfold Join_env.in_terms_in_all_joined_envs.
  change (Join_env.joined_envs (rename_join_env r env))
    with (rename_joined_envs r (Join_env.joined_envs env)).
  rewrite joined_envs_keys_renamed.
  induction (Joined_envs.keys (Join_env.joined_envs env))
    as [ | index keys IH];
    cbn [forallb]; [ reflexivity | ].
  rewrite IH, get_nth_joined_env_renamed, stands_for_renamed.
  destruct
    (Joined_envs.get_nth_joined_env (Join_env.joined_envs env) index)
    as [tenv | ],
    (Join_env.stands_for env index var) as [simple | ];
    cbn [option_map]; try reflexivity.
  rewrite has_alias_in_terms_renamed. reflexivity.
Qed.

Theorem compiler_choices_related : compiler_choices_related_statement.
Proof.
  intros r env var types.
  cbn [Join_env.compiler_choices Join_env.recover_relations_on].
  exact
    (exists_in_all_joined_envs_renamed r (rename_type r)
       (Join_env.joined_envs env) types).
Qed.

Theorem repaired_choices_related : repaired_choices_related_statement.
Proof.
  intros r env var types.
  cbn [Join_env.repaired_choices Join_env.recover_relations_on].
  rewrite in_terms_in_all_joined_envs_renamed.
  f_equal.
  exact
    (exists_in_all_joined_envs_renamed r (rename_type r)
       (Join_env.joined_envs env) types).
Qed.

(* ======================================================================== *)
(* The aliases of the levels                                                *)
(* ======================================================================== *)

Lemma compact_renamed_keys {A B : Type} (r : renaming) (g : A -> B)
    (bindings : finite_map A) :
  compact (Keyed.map_bindings (rename_variable r) g bindings)
  = Keyed.map_bindings (rename_variable r) g (compact bindings).
Proof.
  exact
    (compact_keys_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r)
       g bindings).
Qed.

Lemma is_empty_map_bindings {K A B : Type} (f : K -> K) (g : A -> B)
    (bindings : Keyed.finite_map K A) :
  Keyed.is_empty (Keyed.map_bindings f g bindings) = Keyed.is_empty bindings.
Proof. unfold Keyed.map_bindings. apply is_empty_map. Qed.

Lemma names_with_equations_everywhere_renamed (r : renaming)
    (envs : Joined_envs.t) :
  Join_env.names_with_equations_everywhere (rename_joined_envs r envs)
  = map (rename_name r) (Join_env.names_with_equations_everywhere envs).
Proof.
  unfold Join_env.names_with_equations_everywhere, rename_joined_envs.
  rewrite compact_map_values.
  destruct (Keyed.compact Index.equal envs)
    as [ | [index [tenv equations]] others];
    cbn [Keyed.map_values map fst snd]; [ reflexivity | ].
  rewrite (domain_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  apply filter_renamed. intros name.
  apply forallb_renamed. intros [index' [tenv' equations']]. cbn [fst snd].
  apply (is_bound_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
Qed.

Lemma types_of_name_in_levels_renamed (r : renaming) (envs : Joined_envs.t)
    (name : name) :
  Join_env.types_of_name_in_levels (rename_joined_envs r envs)
    (rename_name r name)
  = Keyed.map_values (rename_type r)
      (Join_env.types_of_name_in_levels envs name).
Proof.
  unfold Join_env.types_of_name_in_levels, rename_joined_envs.
  rewrite compact_map_values. unfold Keyed.map_values.
  apply flat_map_renamed. intros [index [tenv equations]]. cbn [fst snd].
  rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  destruct (Keyed.lookup name_eqb name equations); reflexivity.
Qed.

Lemma join_aliases_into_bindings_renamed (r : renaming)
    (p p' : Join_env.parameters) (fuel : nat) (env : Join_env.t)
    (equations_to_join : Joined_envs.t) :
  parameters_related r p p' ->
  Join_env.join_aliases_into_bindings p' fuel (rename_join_env r env)
    (rename_joined_envs r equations_to_join)
  = option_map (rename_join_env r)
      (Join_env.join_aliases_into_bindings p fuel env equations_to_join).
Proof.
  intros Hp. unfold Join_env.join_aliases_into_bindings. cbv zeta.
  rewrite names_with_equations_everywhere_renamed.
  change (Some (rename_join_env r env))
    with (option_map (rename_join_env r) (Some env)).
  apply fold_left_renamed. intros [env' | ] name; [ | reflexivity ].
  destruct name as [var | sym]; cbn [option_map rename_name];
    [ | reflexivity ].
  change (Join_env.bindings (rename_join_env r env))
    with (rename_bindings_in_target_env r (Join_env.bindings env)).
  rewrite exists_in_target_env_renamed.
  destruct
    (Bindings_in_target_env.exists_in_target_env (Join_env.bindings env) var);
    [ | reflexivity ].
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  rewrite types_of_name_in_levels_renamed.
  apply import_or_delay_n_way_join_type_renamed, Hp.
Qed.

(* ======================================================================== *)
(* Inverse relations                                                        *)
(* ======================================================================== *)

(* For a variable that holds the result of a relation, the relations
   and the names that they are of: the variable and the names are
   renamed. *)
Definition rename_inverse_relations (r : renaming)
    (relations : Join_env.inverse_relations) : Join_env.inverse_relations :=
  Keyed.map_bindings (rename_variable r)
    (map
       (fun relation : Relation.t * name =>
          (fst relation, rename_name r (snd relation))))
    relations.

Lemma add_to_inverse_relations_renamed (r : renaming)
    (relations : Join_env.inverse_relations) (var : variable)
    (relation : Relation.t) (scrutinee : name) :
  Join_env.add_to_inverse_relations (rename_inverse_relations r relations)
    (rename_variable r var) relation (rename_name r scrutinee)
  = rename_inverse_relations r
      (Join_env.add_to_inverse_relations relations var relation scrutinee).
Proof.
  unfold Join_env.add_to_inverse_relations, rename_inverse_relations.
  rewrite lookup_renamed_keys.
  destruct (lookup var relations) as [existing | ]; reflexivity.
Qed.

Lemma with_inverse_relation_renamed (r : renaming)
    (head : head_of_kind_naked_immediate) (relation : Relation.t)
    (scrutinee : name) :
  Join_env.with_inverse_relation (rename_head_of_kind_naked_immediate r head)
    relation (rename_name r scrutinee)
  = rename_type r (Join_env.with_inverse_relation head relation scrutinee).
Proof. reflexivity. Qed.

Lemma add_or_replace_equation_renamed (r : renaming)
    (extension : env_extension) (name : name) (ty : flambda_type) :
  Join_env.add_or_replace_equation (rename_env_extension r extension)
    (rename_name r name) (rename_type r ty)
  = rename_env_extension r
      (Join_env.add_or_replace_equation extension name ty).
Proof.
  destruct extension as [equations].
  unfold Join_env.add_or_replace_equation.
  cbn [rename_env_extension map fst snd]. f_equal. f_equal.
  exact
    (unbind_renamed name_eqb (rename_name r) (name_eqb_renamed r)
       (rename_type r) name equations).
Qed.

Lemma equations_of_renamed (r : renaming) (extension : env_extension) :
  Join_env.equations_of (rename_env_extension r extension)
  = Keyed.map_bindings (rename_name r) (rename_type r)
      (Join_env.equations_of extension).
Proof. destruct extension as [equations]. reflexivity. Qed.

(* The extension in which a name gets a head with one more inverse
   relation. *)
Lemma add_inverse_relation_renamed (r : renaming) (extension : env_extension)
    (name : name) (head : head_of_kind_naked_immediate)
    (relation : Relation.t) (scrutinee : FlambdaTypeGrammar.name) :
  Some
    (Join_env.add_or_replace_equation (rename_env_extension r extension)
       (rename_name r name)
       (Join_env.with_inverse_relation
          (rename_head_of_kind_naked_immediate r head) relation
          (rename_name r scrutinee)))
  = option_map (rename_env_extension r)
      (Some
         (Join_env.add_or_replace_equation extension name
            (Join_env.with_inverse_relation head relation scrutinee))).
Proof.
  cbn [option_map]. f_equal.
  exact
    (add_or_replace_equation_renamed r extension name
       (Join_env.with_inverse_relation head relation scrutinee)).
Qed.

Lemma add_inverse_relation_to_env_extension_renamed (r : renaming)
    (relation : Relation.t) (scrutinee : name) (fuel : nat) :
  ∀ (seen : list name) (extension : env_extension) (name : name),
    Join_env.add_inverse_relation_to_env_extension fuel
      (map (rename_name r) seen) (rename_env_extension r extension)
      (rename_name r name) relation (rename_name r scrutinee)
    = option_map (rename_env_extension r)
        (Join_env.add_inverse_relation_to_env_extension fuel seen extension
           name relation scrutinee).
Proof.
  induction fuel as [ | fuel IH]; intros seen extension name;
    cbn [Join_env.add_inverse_relation_to_env_extension];
    rewrite equations_of_renamed,
      (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r));
    (destruct (Keyed.lookup name_eqb name (Join_env.equations_of extension))
       as [[ty | [ | [[imms relations] | simple] | ] | k ty | ty | ty] | ]);
    cbn [option_map rename_type rename_descr or_unknown_or_bottom_map
         rename_head_of_kind_naked_immediate naked_immediates];
    try reflexivity;
    try
      exact
        (add_inverse_relation_renamed r extension name
           Join_env.empty_naked_immediate_head relation scrutinee);
    try
      (destruct imms as [[ | imm imms] | ];
       [ reflexivity
       | exact
           (add_inverse_relation_renamed r extension name
              (Head_of_kind_naked_immediate (Or_unknown.Known (imm :: imms))
                 relations)
              relation scrutinee)
       | exact
           (add_inverse_relation_renamed r extension name
              (Head_of_kind_naked_immediate Or_unknown.Unknown relations)
              relation scrutinee) ]);
    rewrite name_of_simple_renamed;
    (destruct (name_of_simple simple) as [name' | ]);
    cbn [option_map]; try reflexivity.
  - exact
      (add_inverse_relation_renamed r extension name
         Join_env.empty_naked_immediate_head relation scrutinee).
  - rewrite (among_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
    destruct (Keyed.among name_eqb name' seen);
      [ exact
          (add_inverse_relation_renamed r extension name
             Join_env.empty_naked_immediate_head relation scrutinee)
      | exact (IH (name :: seen) extension name') ].
Qed.

(* The inverse relation for the tag, as [recover_inverse_relations]
   adds it to the extension for blocks. *)
Lemma add_get_tag_relation_renamed (r : renaming) (name : name)
    (get_tag_var : variable) (when_block : env_extension) :
  Join_env.add_inverse_relation_to_env_extension
    (length (Join_env.equations_of (rename_env_extension r when_block))) []
    (rename_env_extension r when_block)
    (Name_var (rename_variable r get_tag_var)) Relation.Get_tag
    (rename_name r name)
  = option_map (rename_env_extension r)
      (Join_env.add_inverse_relation_to_env_extension
         (length (Join_env.equations_of when_block)) [] when_block
         (Name_var get_tag_var) Relation.Get_tag name).
Proof.
  rewrite equations_of_renamed. unfold Keyed.map_bindings.
  rewrite length_map.
  exact
    (add_inverse_relation_to_env_extension_renamed r Relation.Get_tag name
       (length (Join_env.equations_of when_block)) [] when_block
       (Name_var get_tag_var)).
Qed.

(* The type and the relations that [recover_inverse_relations]
   answers. *)
Definition rename_recovered (r : renaming)
    (result : flambda_type * Join_env.inverse_relations)
    : flambda_type * Join_env.inverse_relations :=
  (rename_type r (fst result), rename_inverse_relations r (snd result)).

Lemma recover_inverse_relations_renamed (r : renaming)
    (relations : Join_env.inverse_relations) (name : name)
    (ty : flambda_type) :
  Join_env.recover_inverse_relations (rename_inverse_relations r relations)
    (rename_name r name) (rename_type r ty)
  = option_map (rename_recovered r)
      (Join_env.recover_inverse_relations relations name ty).
Proof.
  destruct ty
    as [[ | [[non_null is_null] | simple] | ] | ty | k ty | ty | ty];
    try reflexivity.
  destruct is_null as [ | is_null].
  - destruct non_null
      as [ | [is_int immediates get_tag blocks extensions is_unique
             | alloc_mode | k contents alloc_mode
             | by_function_slot alloc_mode | strs] | ];
      try reflexivity.
    unfold Join_env.recover_inverse_relations.
    cbn [rename_type rename_descr or_unknown_or_bottom_map
         rename_head_of_kind_value rename_head_of_kind_value_non_null
         rename_is_null].
    assert (Hno :
      match or_unknown_map (rename_type r) immediates with
      | Or_unknown.Known imm_ty => is_obviously_bottom imm_ty
      | Or_unknown.Unknown => false
      end
      = match immediates with
        | Or_unknown.Known imm_ty => is_obviously_bottom imm_ty
        | Or_unknown.Unknown => false
        end).
    { destruct immediates as [imm_ty | ]; cbn [or_unknown_map];
        [ apply is_obviously_bottom_renamed | reflexivity ]. }
    cbv zeta. rewrite Hno. clear Hno.
    destruct
      (match immediates with
       | Or_unknown.Known imm_ty => is_obviously_bottom imm_ty
       | Or_unknown.Unknown => false
       end),
      get_tag as [get_tag_var | ], is_int as [is_int_var | ];
      cbn [option_map rename_recovered fst snd];
      rewrite ?add_to_inverse_relations_renamed; try reflexivity.
    all: destruct extensions as [ | when_immediate when_block].
    all: try rewrite add_get_tag_relation_renamed.
    all:
      try
        (pose proof
           (add_get_tag_relation_renamed r name get_tag_var
              empty_env_extension)
           as Hempty;
         change (rename_env_extension r empty_env_extension)
           with empty_env_extension in Hempty;
         rewrite Hempty; clear Hempty).
    all:
      match goal with
      | |- context
             [Join_env.add_inverse_relation_to_env_extension ?fuel [] ?ext
                (Name_var ?var) Relation.Get_tag ?scrutinee] =>
        destruct
          (Join_env.add_inverse_relation_to_env_extension fuel [] ext
             (Name_var var) Relation.Get_tag scrutinee)
      end;
      reflexivity.
  - unfold Join_env.recover_inverse_relations.
    destruct non_null as [ | head | ], is_null as [is_null_var | ];
      cbn [rename_type rename_descr or_unknown_or_bottom_map
           rename_head_of_kind_value rename_is_null option_map
           rename_recovered fst snd];
      rewrite ?add_to_inverse_relations_renamed; reflexivity.
Qed.

(* ======================================================================== *)
(* The rounds                                                               *)
(* ======================================================================== *)

(* The relations and the state that a round answers. *)
Definition rename_round_result (r : renaming)
    (result : Join_env.inverse_relations * Join_env.t)
    : Join_env.inverse_relations * Join_env.t :=
  (rename_inverse_relations r (fst result), rename_join_env r (snd result)).

Lemma n_way_join_round_renamed (r : renaming)
    (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type) (env : Join_env.t)
    (equations_to_join : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations : Join_env.inverse_relations) :
  choices_related r choices choices' ->
  n_way_join_type_related r join_type join_type' ->
  Join_env.n_way_join_round choices' join_type' (rename_join_env r env)
    (Keyed.map_bindings (rename_variable r)
       (Keyed.map_values (rename_type r)) equations_to_join)
    (rename_inverse_relations r relations)
  = option_map (rename_round_result r)
      (Join_env.n_way_join_round choices join_type env equations_to_join
         relations).
Proof.
  intros Hchoices Hjoin. unfold Join_env.n_way_join_round.
  rewrite compact_renamed_keys.
  change (Some (rename_inverse_relations r relations, rename_join_env r env))
    with (option_map (rename_round_result r) (Some (relations, env))).
  unfold Keyed.map_bindings.
  apply fold_left_renamed.
  intros [[relations' env'] | ] [var types]; [ | reflexivity ].
  cbn [option_map rename_round_result fst snd].
  rewrite bindings_renamed, Hjoin.
  destruct (join_type env' (Index.bindings types)) as [[[ty | ] env''] | ];
    cbn [option_map fst snd or_unknown_map]; try reflexivity.
  rewrite Hchoices.
  destruct (Join_env.recover_relations_on choices env'' var types).
  - change (Name_var (rename_variable r var))
      with (rename_name r (Name_var var)).
    rewrite recover_inverse_relations_renamed.
    destruct
      (Join_env.recover_inverse_relations relations' (Name_var var) ty)
      as [[ty' relations''] | ];
      cbn [option_map rename_recovered fst snd]; [ | reflexivity ].
    rewrite add_equation_in_target_env_renamed.
    destruct (Join_env.add_equation_in_target_env env'' var ty')
      as [env''' | ];
      reflexivity.
  - rewrite add_equation_in_target_env_renamed.
    destruct (Join_env.add_equation_in_target_env env'' var ty)
      as [env''' | ];
      reflexivity.
Qed.

Lemma n_way_join_delayed_equations_renamed (r : renaming)
    (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type) (fuel : nat) :
  choices_related r choices choices' ->
  n_way_join_type_related r join_type join_type' ->
  ∀ (env : Join_env.t) (relations : Join_env.inverse_relations)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type)),
    Join_env.n_way_join_delayed_equations fuel choices' join_type'
      (rename_join_env r env) (rename_inverse_relations r relations)
      (Keyed.map_bindings (rename_variable r)
         (Keyed.map_values (rename_type r)) equations)
    = option_map (rename_round_result r)
        (Join_env.n_way_join_delayed_equations fuel choices join_type env
           relations equations).
Proof.
  intros Hchoices Hjoin.
  induction fuel as [ | fuel IH]; intros env relations equations;
    cbn [Join_env.n_way_join_delayed_equations]; [ reflexivity | ].
  rewrite is_empty_map_bindings.
  destruct (Keyed.is_empty equations); [ reflexivity | ].
  rewrite
    (n_way_join_round_renamed r choices choices' join_type join_type' env
       equations relations Hchoices Hjoin).
  destruct
    (Join_env.n_way_join_round choices join_type env equations relations)
    as [[relations' env'] | ];
    cbn [option_map rename_round_result fst snd]; [ | reflexivity ].
  rewrite new_equations_in_joined_envs_renamed. apply IH.
Qed.

(* The definition of a variable that the join made: the vector that it
   stands for, and its kind. *)
Local Abbreviation rename_definition r :=
  (fun definition : Simples_in_joined_envs.t * flambda_kind =>
     (rename_simples_in_joined_envs r (fst definition), snd definition)).

Lemma equations_of_existentials_renamed (r : renaming)
    (p p' : Join_env.parameters) (env : Join_env.t)
    (existentials :
       list (variable * (Simples_in_joined_envs.t * flambda_kind))) :
  parameters_related r p p' ->
  Join_env.equations_of_existentials p' (rename_join_env r env)
    (Keyed.map_bindings (rename_variable r) (rename_definition r)
       existentials)
  = option_map
      (Keyed.map_bindings (rename_variable r)
         (Keyed.map_values (rename_type r)))
      (Join_env.equations_of_existentials p env existentials).
Proof.
  intros Hp. unfold Keyed.map_bindings.
  induction existentials as [ | [var [simples kind]] existentials IH];
    [ reflexivity | ].
  cbn [map fst snd Join_env.equations_of_existentials] in *.
  change (Join_env.joined_envs (rename_join_env r env))
    with (rename_joined_envs r (Join_env.joined_envs env)).
  rewrite
    (find_simples_in_joined_envs_renamed r (Join_env.expand_const p)
       (Join_env.expand_const p') (Join_env.joined_envs env) simples kind
       (proj1 (proj2 (proj2 (proj2 Hp))))).
  destruct
    (Joined_envs.find_simples_in_joined_envs (Join_env.expand_const p)
       (Join_env.joined_envs env) simples kind)
    as [types | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite IH.
  destruct (Join_env.equations_of_existentials p env existentials)
    as [equations | ];
    reflexivity.
Qed.

Lemma n_way_join_loop_renamed (r : renaming) (p p' : Join_env.parameters)
    (fuel : nat) (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type) (depth_left : nat) :
  parameters_related r p p' ->
  choices_related r choices choices' ->
  n_way_join_type_related r join_type join_type' ->
  ∀ (env : Join_env.t) (relations : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    Join_env.n_way_join_loop depth_left p' fuel choices' join_type'
      (rename_join_env r env) (rename_inverse_relations r relations)
      (Keyed.map_bindings (rename_variable r) (rename_definition r)
         existentials)
    = option_map (rename_round_result r)
        (Join_env.n_way_join_loop depth_left p fuel choices join_type env
           relations existentials).
Proof.
  intros Hp Hchoices Hjoin.
  induction depth_left as [ | depth_left IH];
    intros env relations existentials; cbn [Join_env.n_way_join_loop];
    [ reflexivity | ].
  rewrite is_empty_map_bindings.
  destruct (Keyed.is_empty existentials); [ reflexivity | ].
  rewrite compact_renamed_keys,
    (equations_of_existentials_renamed r p p' env (compact existentials) Hp).
  destruct (Join_env.equations_of_existentials p env (compact existentials))
    as [equations | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite
    (n_way_join_delayed_equations_renamed r choices choices' join_type
       join_type' fuel Hchoices Hjoin env relations equations).
  destruct
    (Join_env.n_way_join_delayed_equations fuel choices join_type env
       relations equations)
    as [[relations' env'] | ];
    cbn [option_map rename_round_result fst snd]; [ | reflexivity ].
  rewrite new_definitions_of_existentials_renamed. apply IH.
Qed.

(* ======================================================================== *)
(* What the join of types calls                                             *)
(* ======================================================================== *)

Lemma joined_env_renamed (r : renaming) (env : Join_env.t) (index : Index.t) :
  Join_env.joined_env (rename_join_env r env) index
  = option_map (rename_typing_env r) (Join_env.joined_env env index).
Proof. apply get_nth_joined_env_renamed. Qed.

Lemma import_type_renamed (r : renaming) (p p' : Join_env.parameters)
    (fuel : nat) (env : Join_env.t) (ty : flambda_type) :
  parameters_related r p p' ->
  Join_env.import_type p' fuel (rename_join_env r env) (rename_type r ty)
  = option_map (rename_join_env r) (Join_env.import_type p fuel env ty).
Proof. apply import_type_transitive_renamed. Qed.

Lemma n_way_join_simples_renamed (r : renaming) (p p' : Join_env.parameters)
    (fuel : nat) (env : Join_env.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) :
  parameters_related r p p' ->
  Join_env.n_way_join_simples p' fuel (rename_join_env r env) kind
    (Keyed.map_values (rename_simple r) simples)
  = option_map
      (fun result : Or_bottom.t simple * Join_env.t =>
         (or_bottom_map (rename_simple r) (fst result),
          rename_join_env r (snd result)))
      (Join_env.n_way_join_simples p fuel env kind simples).
Proof.
  intros Hp. unfold Join_env.n_way_join_simples.
  destruct simples as [ | binding simples]; [ reflexivity | ].
  remember (Keyed.map_values (rename_simple r) (binding :: simples))
    as renamed eqn:Hrenamed.
  destruct renamed as [ | binding' renamed]; [ discriminate Hrenamed | ].
  cbv iota. rewrite Hrenamed. clear Hrenamed binding' renamed.
  change (Join_env.joined_envs (rename_join_env r env))
    with (rename_joined_envs r (Join_env.joined_envs env)).
  rewrite get_canonical_simples_ignoring_name_mode_renamed.
  destruct
    (Joined_envs.get_canonical_simples_ignoring_name_mode
       (Join_env.joined_envs env) (binding :: simples))
    as [canonicals | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite n_way_join_canonical_simples_renamed.
  destruct (Join_env.n_way_join_canonical_simples env canonicals)
    as [[simple | var | canonicals'] | ];
    cbn [option_map rename_canonical_in_target_env]; try reflexivity.
  - rewrite (import_var_transitive_renamed r p p' fuel Hp env var).
    destruct (Join_env.import_var_transitive p fuel env var) as [env' | ];
      reflexivity.
  - rewrite
      (existential_for_these_canonicals_renamed r p p' env canonicals' kind
         Hp).
    destruct
      (Join_env.existential_for_these_canonicals p env canonicals' kind)
      as [var env'].
    reflexivity.
Qed.

Lemma n_way_join_env_extension_renamed (r : renaming)
    (p p' : Join_env.parameters) (fuel : nat) (env : Join_env.t)
    (extensions : list (Index.t * env_extension)) :
  parameters_related r p p' ->
  Join_env.n_way_join_env_extension p' fuel (rename_join_env r env)
    (Keyed.map_values (rename_env_extension r) extensions)
  = option_map
      (or_bottom_map
         (fun result : env_extension * Join_env.t =>
            (rename_env_extension r (fst result),
             rename_join_env r (snd result))))
      (Join_env.n_way_join_env_extension p fuel env extensions).
Proof.
  intros Hp. unfold Join_env.n_way_join_env_extension.
  destruct extensions as [ | [index first_extension] other_extensions];
    [ reflexivity | ].
  cbn [Keyed.map_values map fst snd].
  rewrite
    (forallb_renamed
       (fun binding : Index.t * env_extension =>
          (fst binding, rename_env_extension r (snd binding)))
       (fun binding : Index.t * env_extension =>
          Join_env.env_extension_eqb p (snd binding) first_extension));
    [ | intros [index' extension]; cbn [fst snd];
        apply (proj1 (proj2 (proj2 Hp))) ].
  destruct
    (forallb
       (fun binding : Index.t * env_extension =>
          Join_env.env_extension_eqb p (snd binding) first_extension)
       other_extensions);
    [ | reflexivity ].
  rewrite equations_of_renamed.
  change (Some (rename_join_env r env))
    with (option_map (rename_join_env r) (Some env)).
  unfold Keyed.map_bindings.
  rewrite
    (fold_left_renamed (option_map (rename_join_env r))
       (fun equation : name * flambda_type =>
          (rename_name r (fst equation), rename_type r (snd equation)))
       (fun (env : option Join_env.t) (equation : name * flambda_type) =>
          let* env := env in
          let* env :=
            Join_env.import_name_transitive p fuel env (fst equation)
          in
          Join_env.import_type_transitive p fuel env (snd equation))).
  - destruct
      (List.fold_left
         (fun (env : option Join_env.t) (equation : name * flambda_type) =>
            let* env := env in
            let* env :=
              Join_env.import_name_transitive p fuel env (fst equation)
            in
            Join_env.import_type_transitive p fuel env (snd equation))
         (Join_env.equations_of first_extension) (Some env))
      as [env' | ];
      reflexivity.
  - intros [env' | ] [name ty]; cbn [option_map fst snd]; [ | reflexivity ].
    rewrite (import_name_transitive_renamed r p p' fuel env' name Hp).
    destruct (Join_env.import_name_transitive p fuel env' name)
      as [env'' | ];
      cbn [option_map]; [ | reflexivity ].
    apply import_type_transitive_renamed, Hp.
Qed.

(* ======================================================================== *)
(* Defining the variables of the target                                     *)
(* ======================================================================== *)

Lemma unbind_renamed_keys {A B : Type} (r : renaming) (g : A -> B)
    (var : variable) (bindings : finite_map A) :
  unbind (rename_variable r var)
    (Keyed.map_bindings (rename_variable r) g bindings)
  = Keyed.map_bindings (rename_variable r) g (unbind var bindings).
Proof.
  exact
    (unbind_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r) g var
       bindings).
Qed.

Lemma domain_renamed_keys {A B : Type} (r : renaming) (g : A -> B)
    (bindings : finite_map A) :
  domain (Keyed.map_bindings (rename_variable r) g bindings)
  = map (rename_variable r) (domain bindings).
Proof.
  exact
    (domain_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r) g
       bindings).
Qed.

Lemma move_definition_renamed (r : renaming)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (from to_ : variable) :
  Join_env.move_definition
    (Keyed.map_bindings (rename_variable r) (rename_definition r)
       definitions)
    (rename_variable r from) (rename_variable r to_)
  = Keyed.map_bindings (rename_variable r) (rename_definition r)
      (Join_env.move_definition definitions from to_).
Proof.
  unfold Join_env.move_definition.
  rewrite lookup_renamed_keys.
  destruct (lookup from definitions) as [defn | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite unbind_renamed_keys. reflexivity.
Qed.

Lemma move_equation_renamed (r : renaming)
    (equations : finite_map flambda_type) (from to_ : variable) :
  Join_env.move_equation
    (Keyed.map_bindings (rename_variable r) (rename_type r) equations)
    (rename_variable r from) (rename_variable r to_)
  = option_map (Keyed.map_bindings (rename_variable r) (rename_type r))
      (Join_env.move_equation equations from to_).
Proof.
  unfold Join_env.move_equation.
  rewrite is_bound_renamed_keys.
  destruct (is_bound to_ equations); [ reflexivity | ].
  rewrite lookup_renamed_keys.
  destruct (lookup from equations) as [ty | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite unbind_renamed_keys. reflexivity.
Qed.

Lemma move_inverse_relation_renamed (r : renaming)
    (relations : Join_env.inverse_relations) (from to_ : variable) :
  Join_env.move_inverse_relation (rename_inverse_relations r relations)
    (rename_variable r from) (rename_variable r to_)
  = rename_inverse_relations r
      (Join_env.move_inverse_relation relations from to_).
Proof.
  unfold Join_env.move_inverse_relation, rename_inverse_relations.
  rewrite lookup_renamed_keys.
  destruct (lookup from relations) as [names | ]; cbn [option_map];
    [ | reflexivity ].
  cbv zeta. rewrite unbind_renamed_keys, lookup_renamed_keys.
  destruct (lookup to_ (unbind from relations)) as [existing | ];
    cbn [option_map]; unfold Keyed.map_bindings; cbn [map fst snd];
    rewrite map_app; reflexivity.
Qed.

Lemma in_inverse_relations_renamed (r : renaming)
    (relations : Join_env.inverse_relations) (var : variable) :
  Join_env.in_inverse_relations (rename_inverse_relations r relations)
    (rename_variable r var)
  = Join_env.in_inverse_relations relations var.
Proof.
  unfold Join_env.in_inverse_relations, rename_inverse_relations.
  rewrite is_bound_renamed_keys, compact_renamed_keys. f_equal.
  unfold Keyed.map_bindings. apply existsb_renamed. intros [var' names].
  cbn [fst snd]. apply existsb_renamed. intros [relation name].
  cbn [fst snd].
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  apply name_eqb_renamed.
Qed.

Lemma count_occurrences_renamed (r : renaming) (var : variable)
    (vars : list variable) :
  Join_env.count_occurrences (rename_variable r var)
    (map (rename_variable r) vars)
  = Join_env.count_occurrences var vars.
Proof.
  unfold Join_env.count_occurrences.
  rewrite
    (filter_renamed (rename_variable r) (Pos.eqb var)
       (Pos.eqb (rename_variable r var)))
    by (intros var'; apply rename_variable_eqb).
  apply length_map.
Qed.

Lemma earliest_bound_renamed (r : renaming) (tenv : Typing_env.t)
    (candidates : list variable) :
  ∀ best : variable,
    Join_env.earliest_bound (rename_typing_env r tenv)
      (rename_variable r best) (map (rename_variable r) candidates)
    = option_map (rename_variable r)
        (Join_env.earliest_bound tenv best candidates).
Proof.
  induction candidates as [ | var candidates IH]; intros best;
    cbn [map Join_env.earliest_bound]; [ reflexivity | ].
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  change (Name_var (rename_variable r best))
    with (rename_name r (Name_var best)).
  rewrite !binding_time_and_mode_equivariant.
  destruct (Typing_env.binding_time_and_mode tenv (Name_var var))
    as [time | ];
    [ | reflexivity ].
  destruct (Typing_env.binding_time_and_mode tenv (Name_var best))
    as [best_time | ];
    [ | reflexivity ].
  destruct
    (Binding_time.strictly_earlier
       (Binding_time.With_name_mode.binding_time time)
       (Binding_time.With_name_mode.binding_time best_time));
    apply IH.
Qed.

Lemma choose_alias_renamed (r : renaming) (tenv : Typing_env.t)
    (aliases : list variable) :
  Join_env.choose_alias (rename_typing_env r tenv)
    (map (rename_variable r) aliases)
  = option_map (option_map (rename_variable r))
      (Join_env.choose_alias tenv aliases).
Proof.
  destruct aliases as [ | var aliases]; unfold Join_env.choose_alias;
    cbn [map]; [ reflexivity | ].
  rewrite earliest_bound_renamed.
  destruct (Join_env.earliest_bound tenv var aliases); reflexivity.
Qed.

Lemma expand_variable_renamed (r : renaming)
    (to_expand : finite_map flambda_type) (fuel : nat) :
  ∀ var : variable,
    Join_env.expand_variable fuel
      (Keyed.map_bindings (rename_variable r) (rename_type r) to_expand)
      (rename_variable r var)
    = option_map (rename_type r) (Join_env.expand_variable fuel to_expand var).
Proof.
  induction fuel as [ | fuel IH]; intros var;
    cbn [Join_env.expand_variable]; [ reflexivity | ].
  rewrite lookup_renamed_keys.
  destruct (lookup var to_expand) as [ty | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite get_alias_opt_renamed.
  destruct (get_alias_opt ty) as [[var' | sym | c] | ];
    cbn [option_map rename_simple]; try reflexivity.
  - rewrite domain_renamed_keys, among_renamed_variables.
    destruct (among var' (domain to_expand)); [ apply IH | reflexivity ].
  - rewrite domain_renamed_keys.
    apply project_variables_out_renamed. exact IH.
Qed.

(* The state of the fold over the variables made: the target, the
   types to put for the variables removed, the equations, the inverse
   relations and the definitions. *)
Definition rename_elimination {meet_env : Type} (r : renaming)
    (rename_env : meet_env -> meet_env)
    (state :
       meet_env * finite_map flambda_type * finite_map flambda_type
       * Join_env.inverse_relations
       * finite_map (Simples_in_joined_envs.t * flambda_kind))
    : meet_env * finite_map flambda_type * finite_map flambda_type
      * Join_env.inverse_relations
      * finite_map (Simples_in_joined_envs.t * flambda_kind) :=
  let '(target_env, to_expand, equations, relations, definitions) := state in
  (rename_env target_env,
   Keyed.map_bindings (rename_variable r) (rename_type r) to_expand,
   Keyed.map_bindings (rename_variable r) (rename_type r) equations,
   rename_inverse_relations r relations,
   Keyed.map_bindings (rename_variable r) (rename_definition r) definitions).

(* The target and the definitions that are left. *)
Definition rename_defined {meet_env : Type} (r : renaming)
    (rename_env : meet_env -> meet_env)
    (result :
       meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind))
    : meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind) :=
  (rename_env (fst result),
   Keyed.map_bindings (rename_variable r) (rename_definition r)
     (snd result)).

Lemma define_or_eliminate_variables_and_add_equations_renamed
    {meet_env meet_expanded_head : Type} (r : renaming)
    (rename_env : meet_env -> meet_env)
    (heads_related : meet_expanded_head -> meet_expanded_head -> Prop)
    (ops ops' : Join_env.meet_env_functions meet_env meet_expanded_head)
    (fuel : nat) (m m' : meet_expanded_head) (env : Join_env.t)
    (source_env : meet_env) (relations : Join_env.inverse_relations) :
  meet_env_functions_related r rename_env heads_related ops ops' ->
  heads_related m m' ->
  Join_env.define_or_eliminate_variables_and_add_equations ops' fuel m'
    (rename_join_env r env) (rename_env source_env)
    (rename_inverse_relations r relations)
  = option_map (rename_defined r rename_env)
      (Join_env.define_or_eliminate_variables_and_add_equations ops fuel m
         env source_env relations).
Proof.
  intros
    (Hcreate & Hcurrent & Hadd_var & Hadd_ext & Hbottom & Hfinal & Hnames
     & Hwith)
    Hm.
  unfold Join_env.define_or_eliminate_variables_and_add_equations.
  cbv zeta.
  change (Join_env.bindings (rename_join_env r env))
    with (rename_bindings_in_target_env r (Join_env.bindings env)).
  change (Join_env.types_in_target_env (rename_join_env r env))
    with (Keyed.map_bindings (rename_variable r) (rename_type r)
            (Join_env.types_in_target_env env)).
  change (Join_env.aliases_of_existentials (rename_join_env r env))
    with (rename_aliases_of_existentials r
            (Join_env.aliases_of_existentials env)).
  change (Join_env.definitions_of_existentials (rename_join_env r env))
    with (Keyed.map_bindings (rename_variable r) (rename_definition r)
            (Join_env.definitions_of_existentials env)).
  rewrite imported_variables_in_order_renamed,
    existential_variables_in_order_renamed.
  (* The imported variables are defined. *)
  match goal with
  | |- context
         [List.fold_left ?f
            (Bindings_in_target_env.imported_variables_in_order
               (Join_env.bindings env))
            source_env] =>
    set
      (target_env :=
         List.fold_left f
           (Bindings_in_target_env.imported_variables_in_order
              (Join_env.bindings env))
           source_env)
  end.
  match goal with
  | |- context
         [List.fold_left ?f' (rename_keys r ?l) (rename_env source_env)] =>
    replace (List.fold_left f' (rename_keys r l) (rename_env source_env))
      with (rename_env target_env)
      by (symmetry; subst target_env;
          unfold rename_keys, Keyed.map_bindings; apply fold_left_renamed;
          intros b [var kind]; cbn [fst snd]; apply Hadd_var)
  end.
  (* The variables that the equations mention. *)
  rewrite compact_renamed_keys.
  match goal with
  | |- context [flat_map ?h (compact (Join_env.types_in_target_env env))] =>
    set
      (free_vars :=
         flat_map h (compact (Join_env.types_in_target_env env)))
  end.
  match goal with
  | |- context
         [flat_map ?h'
            (Keyed.map_bindings (rename_variable r) (rename_type r)
               (compact (Join_env.types_in_target_env env)))] =>
    replace
      (flat_map h'
         (Keyed.map_bindings (rename_variable r) (rename_type r)
            (compact (Join_env.types_in_target_env env))))
      with (map (rename_variable r) free_vars)
      by (symmetry; subst free_vars; unfold Keyed.map_bindings;
          apply flat_map_renamed; intros [var ty]; cbn [fst snd];
          apply (proj1 free_variables_rename_type))
  end.
  rewrite Hcurrent.
  (* The variables made are replaced, removed or defined. *)
  match goal with
  | |- context
         [List.fold_left ?f
            (Bindings_in_target_env.existential_variables_in_order
               (Join_env.bindings env))
            ?init] =>
    set
      (eliminated :=
         List.fold_left f
           (Bindings_in_target_env.existential_variables_in_order
              (Join_env.bindings env))
           init)
  end.
  match goal with
  | |- context [List.fold_left ?f' (rename_keys r ?l) ?init'] =>
    replace (List.fold_left f' (rename_keys r l) init')
      with (option_map (rename_elimination r rename_env) eliminated)
  end.
  2: {
    symmetry. subst eliminated. unfold rename_keys, Keyed.map_bindings.
    match goal with
    | |- List.fold_left _ _ ?init' = _ =>
      change init'
        with (option_map (rename_elimination r rename_env)
                (Some
                   (target_env, [], Join_env.types_in_target_env env,
                    relations, Join_env.definitions_of_existentials env)))
    end.
    apply fold_left_renamed.
    intros [[[[[target to_expand] equations] relations'] definitions] | ]
      [var var_kind];
      [ | reflexivity ].
    cbn [option_map rename_elimination fst snd].
    rewrite aliases_of_existential_var_renamed, choose_alias_renamed.
    destruct
      (Join_env.choose_alias (Join_env.current_typing_env ops target_env)
         (Aliases_of_existentials.aliases_of_existential_var
            (Join_env.aliases_of_existentials env) var))
      as [[canon_var | ] | ];
      cbn [option_map]; [ | | reflexivity ].
    - rewrite move_definition_renamed, move_equation_renamed,
        move_inverse_relation_renamed.
      destruct (Join_env.move_equation equations var canon_var)
        as [equations' | ];
        cbn [option_map]; [ | reflexivity ].
      change (Var (rename_variable r canon_var))
        with (rename_simple r (Var canon_var)).
      rewrite <- rename_type_alias_type_of.
      rewrite
        (filter_renamed (rename_variable r)
           (fun alias : variable => negb (Pos.eqb alias canon_var)))
        by (intros alias; rewrite rename_variable_eqb; reflexivity).
      rewrite
        (fold_left_renamed
           (Keyed.map_bindings (rename_variable r) (rename_type r))
           (rename_variable r)
           (fun equations alias =>
              (alias, alias_type_of var_kind (Var canon_var)) :: equations))
        by (intros b c; reflexivity).
      reflexivity.
    - rewrite in_inverse_relations_renamed, count_occurrences_renamed.
      match goal with
      | |- (if ?c then _ else _) = _ => destruct c
      end.
      + rewrite lookup_renamed_keys, !unbind_renamed_keys.
        destruct (lookup var equations) as [ty | ];
          cbn [option_map rename_elimination Keyed.map_bindings map fst snd];
          rewrite ?rename_type_unknown; reflexivity.
      + rewrite Hadd_var. reflexivity.
  }
  destruct eliminated
    as [[[[[target to_expand] equations] relations'] definitions] | ];
    cbn [option_map rename_elimination]; [ | reflexivity ].
  (* The equations, with the types of the variables removed in place. *)
  rewrite domain_renamed_keys, compact_renamed_keys.
  change
    (Keyed.map_bindings (rename_variable r) (rename_type r)
       (compact equations))
    with (map
            (fun binding : variable * flambda_type =>
               (rename_variable r (fst binding), rename_type r (snd binding)))
            (compact equations)).
  match goal with
  | |- context [@traverse ?A ?B ?f (compact equations)] =>
    rewrite
      (@traverse_renamed A A B B
         (fun binding =>
            (rename_variable r (fst binding), rename_type r (snd binding)))
         (fun equation =>
            (rename_name r (fst equation), rename_type r (snd equation)))
         f);
    [ destruct (traverse f (compact equations)) as [equations' | ];
      cbn [option_map]; [ | reflexivity ]
    | intros [var ty] _; cbn [fst snd];
      match goal with
      | |- context
             [project_variables_out (map (rename_variable r) ?tp) ?e' _] =>
        rewrite
          (project_variables_out_renamed r tp
             (Join_env.expand_variable fuel to_expand) e' ty
             (expand_variable_renamed r to_expand fuel))
      end;
      destruct
        (project_variables_out (domain to_expand)
           (Join_env.expand_variable fuel to_expand) ty);
      reflexivity ]
  end.
  (* The equations for the inverse relations. *)
  unfold rename_inverse_relations at 1. rewrite compact_renamed_keys.
  match goal with
  | |- context
         [Keyed.map_bindings (rename_variable r) ?g (compact relations')] =>
    change (Keyed.map_bindings (rename_variable r) g (compact relations'))
      with (map
              (fun binding : variable * list (Relation.t * name) =>
                 (rename_variable r (fst binding), g (snd binding)))
              (compact relations'))
  end.
  match goal with
  | |- context [@traverse ?A ?B ?f (compact relations')] =>
    rewrite
      (@traverse_renamed A A B B
         (fun binding =>
            (rename_variable r (fst binding),
             map
               (fun relation : Relation.t * name =>
                  (fst relation, rename_name r (snd relation)))
               (snd binding)))
         (fun equation =>
            (rename_name r (fst equation), rename_type r (snd equation)))
         f);
    [ destruct (traverse f (compact relations')) as [relation_equations | ];
      cbn [option_map]; [ | reflexivity ]
    | intros [var names] _; cbn [fst snd];
      match goal with
      | |- context
             [project_variables_out (map (rename_variable r) ?tp) ?e' ?t'] =>
        change t'
          with (rename_type r
                  (Type_naked_immediate
                     (Ok
                        (No_alias
                           (Head_of_kind_naked_immediate Or_unknown.Unknown
                              names)))));
        rewrite
          (project_variables_out_renamed r tp
             (Join_env.expand_variable fuel to_expand) e' _
             (expand_variable_renamed r to_expand fuel))
      end;
      destruct
        (project_variables_out (domain to_expand)
           (Join_env.expand_variable fuel to_expand)
           (Type_naked_immediate
              (Ok
                 (No_alias
                    (Head_of_kind_naked_immediate Or_unknown.Unknown names)))));
      reflexivity ]
  end.
  (* The two extensions are added to the target. *)
  match goal with
  | |- context
         [Join_env.add_env_extension ops' (rename_env target) ?ext m'] =>
    change ext with (rename_env_extension r (Env_extension equations'));
    rewrite (Hadd_ext target (Env_extension equations') m m' Hm)
  end.
  destruct
    (Join_env.add_env_extension ops target (Env_extension equations') m)
    as [target' | ];
    cbn [option_map]; [ | reflexivity ].
  match goal with
  | |- context
         [Join_env.add_env_extension ops' (rename_env target') ?ext m'] =>
    change ext with (rename_env_extension r (Env_extension relation_equations));
    rewrite (Hadd_ext target' (Env_extension relation_equations) m m' Hm)
  end.
  destruct
    (Join_env.add_env_extension ops target'
       (Env_extension relation_equations) m)
    as [target'' | ];
    reflexivity.
Qed.

(* ======================================================================== *)
(* Cut and n-way join                                                       *)
(* ======================================================================== *)

Lemma cut_for_joins_renamed (r : renaming) (cut_after : Scope.t)
    (envs : list (Index.t * Typing_env.t)) :
  Join_env.cut_for_joins cut_after
    (Keyed.map_values (rename_typing_env r) envs)
  = option_map (rename_joined_envs r)
      (Join_env.cut_for_joins cut_after envs).
Proof.
  unfold Keyed.map_values, rename_joined_envs.
  induction envs as [ | [index tenv] envs IH]; [ reflexivity | ].
  cbn [map fst snd Join_env.cut_for_joins] in *.
  rewrite IH.
  destruct (Join_env.cut_for_joins cut_after envs) as [rest | ];
    cbn [option_map]; [ | reflexivity ].
  change (Typing_env.is_bottom (rename_typing_env r tenv))
    with (Typing_env.is_bottom tenv).
  destruct (Typing_env.is_bottom tenv); [ reflexivity | ].
  rewrite cut_equivariant.
  destruct (Typing_env.cut tenv cut_after) as [level | ]; reflexivity.
Qed.

Lemma variables_of_env_renamed (r : renaming) (tenv : Typing_env.t) :
  Join_env.variables_of_env (rename_typing_env r tenv)
  = map (rename_variable r) (Join_env.variables_of_env tenv).
Proof.
  unfold Join_env.variables_of_env. rewrite names_to_types_equivariant.
  unfold Keyed.map_bindings. apply flat_map_renamed.
  intros [[var | sym] binding]; reflexivity.
Qed.

(* The target, the definitions that are left and the names of the
   join. *)
Definition rename_join_result {meet_env : Type} (r : renaming)
    (rename_env : meet_env -> meet_env)
    (result :
       meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind)
       * list variable)
    : meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind)
      * list variable :=
  (rename_env (fst (fst result)),
   Keyed.map_bindings (rename_variable r) (rename_definition r)
     (snd (fst result)),
   map (rename_variable r) (snd result)).

Lemma cut_and_n_way_join0_renamed {meet_env meet_expanded_head : Type}
    (r : renaming) (rename_env : meet_env -> meet_env)
    (heads_related : meet_expanded_head -> meet_expanded_head -> Prop)
    (ops ops' : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p p' : Join_env.parameters) (fuel : nat)
    (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type)
    (m m' : meet_expanded_head) (avoid : list variable)
    (source_env : meet_env) (source_tenv : Typing_env.t)
    (equations_to_join : Joined_envs.t) :
  meet_env_functions_related r rename_env heads_related ops ops' ->
  parameters_related r p p' ->
  choices_related r choices choices' ->
  n_way_join_type_related r join_type join_type' ->
  heads_related m m' ->
  Join_env.cut_and_n_way_join0 ops' p' fuel choices' join_type' m'
    (map (rename_variable r) avoid) (rename_env source_env)
    (rename_typing_env r source_tenv)
    (rename_joined_envs r equations_to_join)
  = option_map (rename_join_result r rename_env)
      (Join_env.cut_and_n_way_join0 ops p fuel choices join_type m avoid
         source_env source_tenv equations_to_join).
Proof.
  intros Hops Hp Hchoices Hjoin Hm.
  pose proof Hops
    as (Hcreate & Hcurrent & Hadd_var & Hadd_ext & Hbottom & Hfinal & Hnames
        & Hwith).
  unfold Join_env.cut_and_n_way_join0. cbv zeta.
  assert (Hempty :
    Keyed.is_empty (rename_joined_envs r equations_to_join)
    = Keyed.is_empty equations_to_join)
    by (destruct equations_to_join; reflexivity).
  rewrite Hempty. clear Hempty.
  destruct (Keyed.is_empty equations_to_join).
  { cbn [option_map rename_join_result fst snd]. rewrite Hbottom. reflexivity. }
  rewrite from_source_env_renamed, join_env_create_renamed.
  set
    (empty_env :=
       Join_env.create equations_to_join
         (Bindings_in_target_env.from_source_env avoid source_tenv)).
  rewrite
    (join_aliases_into_bindings_renamed r p p' fuel empty_env
       equations_to_join Hp).
  destruct
    (Join_env.join_aliases_into_bindings p fuel empty_env equations_to_join)
    as [env | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite new_equations_in_joined_envs_renamed.
  match goal with
  | |- context
         [Join_env.n_way_join_delayed_equations fuel choices' join_type'
            (rename_join_env r env) ?relations ?equations] =>
    change
      (Join_env.n_way_join_delayed_equations fuel choices' join_type'
         (rename_join_env r env) relations equations)
      with (Join_env.n_way_join_delayed_equations fuel choices' join_type'
              (rename_join_env r env) (rename_inverse_relations r [])
              equations)
  end.
  rewrite
    (n_way_join_delayed_equations_renamed r choices choices' join_type
       join_type' fuel Hchoices Hjoin env []).
  destruct
    (Join_env.n_way_join_delayed_equations fuel choices join_type env []
       (Join_env.new_equations_in_joined_envs env empty_env))
    as [[relations env'] | ];
    cbn [option_map rename_round_result fst snd]; [ | reflexivity ].
  rewrite new_definitions_of_existentials_renamed,
    (proj2 (proj2 (proj2 (proj2 Hp)))),
    (n_way_join_loop_renamed r p p' fuel choices choices' join_type
       join_type' (Join_env.join_depth p) Hp Hchoices Hjoin env' relations).
  destruct
    (Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices
       join_type env' relations
       (Join_env.new_definitions_of_existentials env' empty_env))
    as [[relations' env''] | ];
    cbn [option_map rename_round_result fst snd]; [ | reflexivity ].
  change (Join_env.bindings (rename_join_env r env''))
    with (rename_bindings_in_target_env r (Join_env.bindings env'')).
  change
    (Bindings_in_target_env.names_to_avoid
       (rename_bindings_in_target_env r (Join_env.bindings env'')))
    with (map (rename_variable r)
            (Bindings_in_target_env.names_to_avoid (Join_env.bindings env''))).
  rewrite Hwith,
    (define_or_eliminate_variables_and_add_equations_renamed r rename_env
       heads_related ops ops' fuel m m' env''
       (Join_env.with_names_to_avoid ops source_env
          (Bindings_in_target_env.names_to_avoid (Join_env.bindings env'')))
       relations' Hops Hm).
  destruct
    (Join_env.define_or_eliminate_variables_and_add_equations ops fuel m
       env''
       (Join_env.with_names_to_avoid ops source_env
          (Bindings_in_target_env.names_to_avoid (Join_env.bindings env'')))
       relations')
    as [[target_env definitions] | ];
    reflexivity.
Qed.

Theorem cut_and_n_way_join_related : cut_and_n_way_join_related_statement.
Proof.
  intros meet_env meet_expanded_head r rename_env heads_related ops ops' p
    p' fuel choices choices' join_type join_type' m m' cut_after source_env
    source_tenv joined_envs Hops Hp Hchoices Hjoin Hm.
  pose proof Hops
    as (Hcreate & Hcurrent & Hadd_var & Hadd_ext & Hbottom & Hfinal & Hnames
        & Hwith).
  unfold Join_env.cut_and_n_way_join. cbv zeta.
  rewrite number_renamed, cut_for_joins_renamed.
  destruct (Join_env.cut_for_joins cut_after (Index.number joined_envs))
    as [equations_to_join | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite Hnames, variables_of_env_renamed,
    (flat_map_renamed (rename_typing_env r) (rename_variable r)
       Join_env.variables_of_env Join_env.variables_of_env)
    by (intros tenv; apply variables_of_env_renamed).
  rewrite <- !map_app.
  rewrite
    (cut_and_n_way_join0_renamed r rename_env heads_related ops ops' p p'
       fuel choices choices' join_type join_type' m m'
       (Join_env.names_to_avoid ops source_env
        ++ Join_env.variables_of_env source_tenv
        ++ flat_map Join_env.variables_of_env joined_envs)
       source_env source_tenv equations_to_join Hops Hp Hchoices Hjoin Hm).
  destruct
    (Join_env.cut_and_n_way_join0 ops p fuel choices join_type m
       (Join_env.names_to_avoid ops source_env
        ++ Join_env.variables_of_env source_tenv
        ++ flat_map Join_env.variables_of_env joined_envs)
       source_env source_tenv equations_to_join)
    as [[[target_env definitions] names] | ];
    reflexivity.
Qed.

(* ======================================================================== *)
(* The join of types of the model                                           *)
(* ======================================================================== *)

Module Meet_parts := FlambdaEnvRenameMeetPartsProofs.

Lemma join_expanded_renamed (r : renaming)
    (expanded : FlambdaExpandHead.Expanded_type.t) :
  join_expanded (FlambdaProversRenameSpec.rename_expanded r expanded)
  = FlambdaProversRenameSpec.rename_expanded r
      (join_expanded expanded).
Proof.
  destruct expanded as [kind [ | [head | head | k head | head | head] | ]];
    reflexivity.
Qed.

Lemma expand_head_for_join_renamed (r : renaming) (tenv : Typing_env.t)
    (t : flambda_type) :
  expand_head_for_join (rename_typing_env r tenv) (rename_type r t)
  = option_map (FlambdaProversRenameSpec.rename_expanded r)
      (expand_head_for_join tenv t).
Proof.
  unfold expand_head_for_join.
  rewrite FlambdaProversRenameProofs.expand_head_rename.
  destruct (FlambdaExpandHead.expand_head tenv t) as [expanded | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite join_expanded_renamed. reflexivity.
Qed.

Theorem n_way_join_of_types_related : n_way_join_of_types_related_statement.
Proof.
  intros r p p' fuel_of_env fuel Hp env types.
  unfold n_way_join_of_types.
  exact
    (FlambdaEnvRenameJoinTypesProofs.n_way_join_renamed r (rename_join_env r)
       Join_env.joined_env Join_env.joined_env
       (Join_env.n_way_join_simples p fuel_of_env)
       (Join_env.n_way_join_simples p' fuel_of_env)
       (Join_env.import_type p fuel_of_env)
       (Join_env.import_type p' fuel_of_env)
       (Join_env.n_way_join_env_extension p fuel_of_env)
       (Join_env.n_way_join_env_extension p' fuel_of_env)
       expand_head_for_join expand_head_for_join
       reduce_head_of_kind_naked_immediate
       reduce_head_of_kind_naked_immediate type_eqb type_eqb block_shape_eqb
       (joined_env_renamed r)
       (fun env kind simples =>
          n_way_join_simples_renamed r p p' fuel_of_env env kind simples Hp)
       (fun env ty => import_type_renamed r p p' fuel_of_env env ty Hp)
       (fun env extensions =>
          n_way_join_env_extension_renamed r p p' fuel_of_env env extensions
            Hp)
       (expand_head_for_join_renamed r)
       (Meet_parts.reduce_head_of_kind_naked_immediate_renamed r)
       (proj1 type_eqb_rename r) fuel env types).
Qed.

(* ======================================================================== *)
(* The join that Simplify calls                                             *)
(* ======================================================================== *)

Lemma simples_at_normal_mode_renamed (r : renaming) (envs : Joined_envs.t)
    (simples : list (Index.t * simple)) :
  Analysis.simples_at_normal_mode (rename_joined_envs r envs)
    (Keyed.map_values (rename_simple r) simples)
  = option_map (option_map (rename_simples_in_joined_envs r))
      (Analysis.simples_at_normal_mode envs simples).
Proof.
  unfold rename_simples_in_joined_envs, Keyed.map_values.
  induction simples as [ | [index simple] simples IH]; [ reflexivity | ].
  cbn [map fst snd Analysis.simples_at_normal_mode] in *.
  rewrite get_nth_joined_env_renamed.
  destruct (Joined_envs.get_nth_joined_env envs index) as [tenv | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite get_canonical_simple_exn_equivariant.
  destruct
    (Typing_env.get_canonical_simple_exn tenv (Some Name_mode.normal) None
       simple)
    as [canonical | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite IH.
  destruct (Analysis.simples_at_normal_mode envs simples) as [rest | ];
    cbn [option_map]; [ | reflexivity ].
  destruct canonical as [canonical | ], rest as [rest | ]; reflexivity.
Qed.

Lemma definitions_at_normal_mode_renamed (r : renaming)
    (envs : Joined_envs.t)
    (definitions :
       list (variable * (Simples_in_joined_envs.t * flambda_kind))) :
  Analysis.definitions_at_normal_mode (rename_joined_envs r envs)
    (Keyed.map_bindings (rename_variable r) (rename_definition r)
       definitions)
  = option_map
      (Keyed.map_bindings (rename_variable r) (rename_definition r))
      (Analysis.definitions_at_normal_mode envs definitions).
Proof.
  unfold Keyed.map_bindings.
  induction definitions as [ | [var [simples kind]] definitions IH];
    [ reflexivity | ].
  cbn [map fst snd Analysis.definitions_at_normal_mode] in *.
  change (rename_simples_in_joined_envs r simples)
    with (Keyed.map_values (rename_simple r) simples).
  rewrite compact_map_values, simples_at_normal_mode_renamed.
  destruct
    (Analysis.simples_at_normal_mode envs
       (Keyed.compact Index.equal simples))
    as [at_normal_mode | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite IH.
  destruct (Analysis.definitions_at_normal_mode envs definitions)
    as [rest | ];
    cbn [option_map]; [ | reflexivity ].
  destruct at_normal_mode as [simples' | ]; reflexivity.
Qed.

Lemma analysis_create_renamed {A : Type} (r : renaming)
    (external_ids : Keyed.finite_map Index.t A) (envs : Joined_envs.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  Analysis.create external_ids (rename_joined_envs r envs)
    (Keyed.map_bindings (rename_variable r) (rename_definition r)
       definitions)
  = option_map (rename_analysis r)
      (Analysis.create external_ids envs definitions).
Proof.
  unfold Analysis.create.
  rewrite compact_renamed_keys, definitions_at_normal_mode_renamed.
  destruct (Analysis.definitions_at_normal_mode envs (compact definitions));
    reflexivity.
Qed.

(* The identifiers of the environments that are not bottom. *)
Lemma external_ids_renamed {A : Type} (r : renaming)
    (numbered : list (Index.t * (A * Typing_env.t))) :
  flat_map
    (fun binding : Index.t * (A * Typing_env.t) =>
       if Typing_env.is_bottom (snd (snd binding))
       then []
       else [(fst binding, fst (snd binding))])
    (Keyed.map_values
       (fun use : A * Typing_env.t =>
          (fst use, rename_typing_env r (snd use)))
       numbered)
  = flat_map
      (fun binding : Index.t * (A * Typing_env.t) =>
         if Typing_env.is_bottom (snd (snd binding))
         then []
         else [(fst binding, fst (snd binding))])
      numbered.
Proof.
  unfold Keyed.map_values.
  induction numbered as [ | [index [id tenv]] numbered IH];
    [ reflexivity | ].
  cbn [map flat_map fst snd]. rewrite IH. reflexivity.
Qed.

Theorem cut_and_n_way_join_with_analysis_related :
  cut_and_n_way_join_with_analysis_related_statement.
Proof.
  intros A meet_env meet_expanded_head r rename_env heads_related ops ops' p
    p' fuel choices choices' join_type join_type' m m' avoid cut_after
    source_tenv joined_envs Hops Hp Hchoices Hjoin Hm.
  pose proof Hops
    as (Hcreate & Hcurrent & Hadd_var & Hadd_ext & Hbottom & Hfinal & Hnames
        & Hwith).
  unfold cut_and_n_way_join_with_analysis. cbv zeta.
  rewrite number_renamed.
  match goal with
  | |- context [flat_map ?h (Keyed.map_values ?g ?numbered)] =>
    replace (flat_map h (Keyed.map_values g numbered))
      with (flat_map h numbered)
      by (symmetry; apply external_ids_renamed)
  end.
  match goal with
  | |- context
         [Join_env.cut_for_joins cut_after
            (map ?proj (Keyed.map_values ?g ?numbered))] =>
    replace (map proj (Keyed.map_values g numbered))
      with (Keyed.map_values (rename_typing_env r) (map proj numbered))
      by (unfold Keyed.map_values; rewrite !map_map; reflexivity)
  end.
  rewrite cut_for_joins_renamed.
  match goal with
  | |- context [match option_map _ ?x with _ => _ end] =>
    destruct x as [equations_to_join | ]
  end;
    cbn [option_map]; [ | reflexivity ].
  rewrite Hcreate,
    (cut_and_n_way_join0_renamed r rename_env heads_related ops ops' p p'
       fuel choices choices' join_type join_type' m m' avoid
       (Join_env.create_meet_env ops source_tenv) source_tenv
       equations_to_join Hops Hp Hchoices Hjoin Hm).
  match goal with
  | |- context [match option_map _ ?x with _ => _ end] =>
    destruct x as [[[target_env definitions] names] | ]
  end;
    cbn [option_map rename_join_result fst snd]; [ | reflexivity ].
  rewrite Hfinal, analysis_create_renamed.
  match goal with
  | |- context [match option_map _ ?x with _ => _ end] => destruct x
  end;
    reflexivity.
Qed.

Theorem join_levels_cut_and_n_way_join_related :
  join_levels_cut_and_n_way_join_related_statement.
Proof.
  intros A meet_env meet_expanded_head r rename_env heads_related ops ops' p
    p' fuel choices choices' join_type join_type' m m' avoid
    definition_typing_env ts_and_use_ids params cut_after extra Hops Hp
    Hchoices Hjoin Hm.
  unfold join_levels_cut_and_n_way_join.
  rewrite
    (forallb_renamed (rename_variable r)
       (fun param : variable =>
          Typing_env.mem None definition_typing_env (Name_var param)))
    by (intros param;
        exact
          (mem_equivariant r None definition_typing_env (Name_var param))).
  assert (Hsymbols :
    forallb
      (fun symbol : symbol =>
         Typing_env.mem None (rename_typing_env r definition_typing_env)
           (Name_symbol symbol))
      extra
    = forallb
        (fun symbol : symbol =>
           Typing_env.mem None definition_typing_env (Name_symbol symbol))
        extra).
  { induction extra as [ | symbol extra IH]; cbn [forallb];
      [ reflexivity | ].
    rewrite IH. f_equal;
      exact
        (mem_equivariant r None definition_typing_env (Name_symbol symbol)). }
  rewrite Hsymbols. clear Hsymbols.
  match goal with
  | |- (if ?c then _ else _) = _ => destruct c
  end;
    [ | reflexivity ].
  match goal with
  | |- context [rev (map ?swap (map ?renamed ts_and_use_ids))] =>
    replace (rev (map swap (map renamed ts_and_use_ids)))
      with (map
              (fun use : A * Typing_env.t =>
                 (fst use, rename_typing_env r (snd use)))
              (rev (map swap ts_and_use_ids)))
      by (rewrite map_rev, !map_map; reflexivity)
  end.
  exact
    (cut_and_n_way_join_with_analysis_related A meet_env meet_expanded_head r
       rename_env heads_related ops ops' p p' fuel choices choices' join_type
       join_type' m m' avoid cut_after definition_typing_env _ Hops Hp
       Hchoices Hjoin Hm).
Qed.

Theorem analysis_read_renamed : analysis_read_renamed_statement.
Proof.
  intros A r. repeat split.
  - intros analysis env simple. unfold Analysis.simple_refined_at_join_of.
    rewrite get_canonical_ignoring_name_mode_equivariant.
    destruct (Typing_env.get_canonical_simple_ignoring_name_mode env simple)
      as [var | sym | c];
      cbn [rename_simple]; try reflexivity.
    cbn [rename_analysis Analysis.definitions_in_joined_envs].
    unfold rename_definitions. rewrite lookup_renamed_keys.
    destruct (lookup var (Analysis.definitions_in_joined_envs analysis))
      as [[canonicals kind] | ];
      reflexivity.
  - intros external_ids canonicals.
    unfold Analysis.values_at_uses, rename_simples_in_joined_envs.
    rewrite compact_map_values. unfold Keyed.map_values.
    induction (Keyed.compact Index.equal canonicals)
      as [ | [index simple] bindings IH];
      [ reflexivity | ].
    cbn [map fst snd Analysis.values_of_bindings]. rewrite IH.
    destruct simple; reflexivity.
  - intros external_ids canonicals.
    unfold Analysis.definitions_at_uses, rename_simples_in_joined_envs.
    rewrite compact_map_values. unfold Keyed.map_values.
    induction (Keyed.compact Index.equal canonicals)
      as [ | [index simple] bindings IH];
      [ reflexivity | ].
    cbn [map fst snd Analysis.definitions_of_bindings]. rewrite IH.
    destruct (Keyed.lookup Index.equal index external_ids)
      as [external_id | ];
      [ | reflexivity ].
    destruct (Analysis.definitions_of_bindings external_ids bindings);
      reflexivity.
  - intros analysis. unfold Analysis.variables_created_at_join.
    cbn [rename_analysis Analysis.canonical_definitions_at_normal_mode].
    unfold rename_definitions.
    rewrite compact_renamed_keys. unfold Keyed.map_bindings.
    rewrite !map_map. apply map_ext. intros [var [simples kind]].
    reflexivity.
Qed.
