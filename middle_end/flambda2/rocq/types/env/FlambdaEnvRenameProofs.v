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


(* Proofs of the statements of FlambdaEnvRenameSpec.v. Nothing here needs
   review.

   The laws of the action are had part by part: a map whose keys and
   values are renamed has each law when the renaming of its keys and
   that of its values have it. The parts are the action on types
   (FlambdaTypeRenameProofs.v) and the action on the aliases
   (FlambdaEnvAliasesRenamingProofs.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaNominalInstancesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaEnvAliasesRenamingSpec FlambdaEnvAliasesRenamingProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaTypeNominalSpec FlambdaTypeNominalProofs FlambdaEnvRenameSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* Maps whose keys and values are renamed                                   *)
(* ======================================================================== *)

Lemma map_bindings_same {K A : Type} (f : K -> K) (g : A -> A)
    (bindings : Keyed.finite_map K A) :
  (∀ (k : K) (a : A), (k, a) ∈ bindings -> f k = k ∧ g a = a) ->
  Keyed.map_bindings f g bindings = bindings.
Proof.
  intros Hsame. unfold Keyed.map_bindings.
  induction bindings as [ | [k a] bindings IH]; simpl; [ reflexivity | ].
  destruct (Hsame k a (or_introl eq_refl)) as [-> ->]. rewrite IH;
    [ reflexivity | ].
  intros k' a' Hin. apply Hsame. right. exact Hin.
Qed.

Lemma map_bindings_alike {K K' A B : Type} (f f' : K -> K') (g g' : A -> B)
    (bindings : Keyed.finite_map K A) :
  (∀ (k : K) (a : A), (k, a) ∈ bindings -> f k = f' k ∧ g a = g' a) ->
  Keyed.map_bindings f g bindings = Keyed.map_bindings f' g' bindings.
Proof.
  intros Halike. unfold Keyed.map_bindings. apply map_ext_in.
  intros [k a] Hin. simpl. destruct (Halike k a Hin) as [-> ->].
  reflexivity.
Qed.

Lemma map_bindings_twice {K K' K'' A B C : Type} (f : K' -> K'')
    (f' : K -> K') (g : B -> C) (g' : A -> B)
    (bindings : Keyed.finite_map K A) :
  Keyed.map_bindings f g (Keyed.map_bindings f' g' bindings)
  = Keyed.map_bindings (fun k => f (f' k)) (fun a => g (g' a)) bindings.
Proof. unfold Keyed.map_bindings. rewrite map_map. reflexivity. Qed.

Lemma in_flat_map_of {A B : Type} (f : A -> list B) (l : list A) (a : A)
    (b : B) :
  a ∈ l -> b ∈ f a -> b ∈ flat_map f l.
Proof. intros Ha Hb. apply in_flat_map. exists a. split; assumption. Qed.

Lemma flat_map_mapped {A B C D : Type} (h : A -> B) (f : A -> list C)
    (f' : B -> list D) (g : C -> D) (l : list A) :
  (∀ a : A, f' (h a) = map g (f a)) ->
  flat_map f' (map h l) = map g (flat_map f l).
Proof.
  intros Hf. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH, map_app. reflexivity.
Qed.

(* ======================================================================== *)
(* The parts: names, and what the environment knows of a name               *)
(* ======================================================================== *)

Lemma rename_variable_identity (x : variable) :
  rename_variable identity_renaming x = x.
Proof. reflexivity. Qed.

Lemma rename_name_same (n : name) : rename_name identity_renaming n = n.
Proof. destruct n; reflexivity. Qed.

Lemma rename_name_twice (second first : renaming) (n : name) :
  rename_name (compose_renaming second first) n
  = rename_name second (rename_name first n).
Proof.
  destruct n; simpl; [ rewrite rename_variable_compose | ]; reflexivity.
Qed.

Lemma rename_name_alike (r r' : renaming) (n : name) :
  (∀ x : variable,
    x ∈ free_variables_of_name n ->
    rename_variable r x = rename_variable r' x) ->
  rename_name r n = rename_name r' n.
Proof.
  destruct n as [x | s]; simpl; intros Halike; [ | reflexivity ].
  rewrite (Halike x (or_introl eq_refl)). reflexivity.
Qed.

Lemma rename_binding_same
    (binding : flambda_type * Binding_time.With_name_mode.t) :
  rename_binding identity_renaming binding = binding.
Proof.
  destruct binding as [ty binding_time]. unfold rename_binding. simpl.
  rewrite (proj1 rename_type_identity). reflexivity.
Qed.

Lemma rename_binding_twice (second first : renaming)
    (binding : flambda_type * Binding_time.With_name_mode.t) :
  rename_binding (compose_renaming second first) binding
  = rename_binding second (rename_binding first binding).
Proof.
  unfold rename_binding. simpl. rewrite (proj1 rename_type_compose).
  reflexivity.
Qed.

Lemma rename_binding_alike (r r' : renaming)
    (binding : flambda_type * Binding_time.With_name_mode.t) :
  (∀ x : variable,
    x ∈ free_variables (fst binding) ->
    rename_variable r x = rename_variable r' x) ->
  rename_binding r binding = rename_binding r' binding.
Proof.
  intros Halike. unfold rename_binding.
  rewrite (proj1 rename_type_extensional r r' (fst binding) Halike).
  reflexivity.
Qed.

(* A map from names whose values are renamed by [g], whose free variables
   are [fv]: the laws, from those of [g]. *)
Section Named_bindings.

Context {B : Type} (g : renaming -> B -> B) (fv : B -> list variable).

Hypothesis g_same : ∀ b : B, g identity_renaming b = b.
Hypothesis g_twice :
  ∀ (second first : renaming) (b : B),
    g (compose_renaming second first) b = g second (g first b).
Hypothesis g_alike :
  ∀ (r r' : renaming) (b : B),
    (∀ x : variable, x ∈ fv b -> rename_variable r x = rename_variable r' x)
    ->
    g r b = g r' b.
Hypothesis g_free :
  ∀ (r : renaming) (b : B), fv (g r b) = map (rename_variable r) (fv b).

Definition bindings_free_variables (bindings : Keyed.finite_map name B)
    : list variable :=
  flat_map
    (fun binding => free_variables_of_name (fst binding) ++ fv (snd binding))
    bindings.

Lemma named_bindings_same (bindings : Keyed.finite_map name B) :
  Keyed.map_bindings (rename_name identity_renaming) (g identity_renaming)
    bindings
  = bindings.
Proof.
  apply map_bindings_same. intros n b _. split;
    [ apply rename_name_same | apply g_same ].
Qed.

Lemma named_bindings_twice (second first : renaming)
    (bindings : Keyed.finite_map name B) :
  Keyed.map_bindings (rename_name (compose_renaming second first))
    (g (compose_renaming second first)) bindings
  = Keyed.map_bindings (rename_name second) (g second)
      (Keyed.map_bindings (rename_name first) (g first) bindings).
Proof.
  rewrite map_bindings_twice. apply map_bindings_alike. intros n b _.
  split; [ apply rename_name_twice | apply g_twice ].
Qed.

Lemma named_bindings_alike (r r' : renaming)
    (bindings : Keyed.finite_map name B) :
  (∀ x : variable,
    x ∈ bindings_free_variables bindings ->
    rename_variable r x = rename_variable r' x) ->
  Keyed.map_bindings (rename_name r) (g r) bindings
  = Keyed.map_bindings (rename_name r') (g r') bindings.
Proof.
  intros Halike. apply map_bindings_alike. intros n b Hin.
  split; [ apply rename_name_alike | apply g_alike ]; intros x Hx;
    apply Halike;
    apply (in_flat_map_of _ bindings (n, b) x Hin), in_or_app;
    [ left | right ]; exact Hx.
Qed.

Lemma named_bindings_free (r : renaming)
    (bindings : Keyed.finite_map name B) :
  bindings_free_variables (Keyed.map_bindings (rename_name r) (g r) bindings)
  = map (rename_variable r) (bindings_free_variables bindings).
Proof.
  unfold bindings_free_variables, Keyed.map_bindings.
  apply flat_map_mapped. intros [n b]. simpl.
  rewrite FlambdaTypeRenameProofs.free_variables_of_name_renamed, g_free,
    map_app.
  reflexivity.
Qed.

End Named_bindings.

(* ======================================================================== *)
(* The aliases: two renamings that agree on the free variables              *)
(* ======================================================================== *)

Lemma rename_aliases_inverse_r (r : renaming) (t : Aliases.t) :
  rename_aliases r (rename_aliases (inverse_renaming r) t) = t.
Proof.
  rewrite <- rename_aliases_compose.
  rewrite (rename_aliases_extensional _ identity_renaming);
    [ apply rename_aliases_identity | ].
  intros x. rewrite rename_variable_compose. apply rename_variable_inverse_r.
Qed.

Lemma rename_aliases_alike (r r' : renaming) (t : Aliases.t) :
  (∀ x : variable,
    x ∈ aliases_free_variables t ->
    rename_variable r x = rename_variable r' x) ->
  rename_aliases r t = rename_aliases r' t.
Proof.
  intros Halike.
  assert (Hfixed :
    rename_aliases (compose_renaming (inverse_renaming r) r') t = t).
  { apply rename_aliases_fixes. intros x Hx.
    rewrite rename_variable_compose, <- (Halike x (proj1 (among_In _ _) Hx)).
    apply rename_variable_inverse_l. }
  rewrite rename_aliases_compose in Hfixed.
  rewrite <- Hfixed at 1. apply rename_aliases_inverse_r.
Qed.

(* ======================================================================== *)
(* Levels                                                                   *)
(* ======================================================================== *)

Lemma level_eq (defined_vars defined_vars' : finite_map flambda_kind)
    (binding_times binding_times'
      : Keyed.finite_map Binding_time.t (list variable))
    (equations equations' : Keyed.finite_map name flambda_type) :
  defined_vars = defined_vars' -> binding_times = binding_times' ->
  equations = equations' ->
  Typing_env_level.create defined_vars binding_times equations
  = Typing_env_level.create defined_vars' binding_times' equations'.
Proof. intros -> -> ->. reflexivity. Qed.

Theorem rename_level_identity : rename_level_identity_statement.
Proof.
  intros [defined_vars binding_times equations]. unfold rename_level. simpl.
  apply level_eq.
  - apply map_bindings_same. intros x kind _. split; reflexivity.
  - unfold Keyed.map_values.
    induction binding_times as [ | [binding_time vars] binding_times IH];
      simpl; [ reflexivity | ].
    rewrite map_id, IH. reflexivity.
  - apply (named_bindings_same rename_type). exact (proj1 rename_type_identity).
Qed.

Theorem rename_level_compose : rename_level_compose_statement.
Proof.
  intros second first [defined_vars binding_times equations].
  unfold rename_level. simpl. apply level_eq.
  - unfold Keyed.map_keys. rewrite map_bindings_twice.
    apply map_bindings_alike. intros x kind _. split; [ | reflexivity ].
    apply rename_variable_compose.
  - unfold Keyed.map_values. rewrite map_map. apply map_ext.
    intros [binding_time vars]. simpl. rewrite map_map. f_equal.
    apply map_ext. intros x. apply rename_variable_compose.
  - apply (named_bindings_twice rename_type).
    exact (proj1 rename_type_compose).
Qed.

Theorem rename_level_extensional : rename_level_extensional_statement.
Proof.
  intros r r' [defined_vars binding_times equations] Halike.
  unfold level_free_variables in Halike. simpl in Halike.
  unfold rename_level. simpl. apply level_eq.
  - apply map_bindings_alike. intros x kind Hin. split; [ | reflexivity ].
    apply Halike, in_or_app. left. exact (in_map fst _ _ Hin).
  - unfold Keyed.map_values. apply map_ext_in.
    intros [binding_time vars] Hin. simpl. f_equal. apply map_ext_in.
    intros x Hx. apply Halike, in_or_app. right. apply in_or_app. left.
    exact (in_flat_map_of snd _ _ x Hin Hx).
  - apply (named_bindings_alike rename_type free_variables).
    + exact (proj1 rename_type_extensional).
    + intros x Hx. apply Halike, in_or_app. right. apply in_or_app. right.
      exact Hx.
Qed.

Theorem rename_level_fixes : rename_level_fixes_statement.
Proof.
  intros r level Hfixes.
  rewrite (rename_level_extensional r identity_renaming level);
    [ apply rename_level_identity | ].
  exact Hfixes.
Qed.

Theorem rename_level_inverse : rename_level_inverse_statement.
Proof.
  intros r level. rewrite <- rename_level_compose.
  apply rename_level_fixes. intros x _.
  rewrite rename_variable_compose. apply rename_variable_inverse_l.
Qed.

Theorem free_variables_rename_level : free_variables_rename_level_statement.
Proof.
  intros r [defined_vars binding_times equations].
  unfold level_free_variables, rename_level. simpl. rewrite !map_app.
  apply f_equal2; [ | apply f_equal2 ].
  - unfold Keyed.map_keys, Keyed.map_bindings. rewrite !map_map. reflexivity.
  - unfold Keyed.map_values. apply flat_map_mapped. intros [bt vars].
    reflexivity.
  - apply (named_bindings_free rename_type free_variables).
    exact (proj1 free_variables_rename_type).
Qed.

(* ======================================================================== *)
(* What the environment knows of its names                                  *)
(* ======================================================================== *)

Lemma cached_level_eq
    (names_to_types names_to_types'
      : Keyed.finite_map name (flambda_type * Binding_time.With_name_mode.t))
    (aliases aliases' : Aliases.t) :
  names_to_types = names_to_types' -> aliases = aliases' ->
  {| Cached_level.names_to_types := names_to_types;
     Cached_level.aliases := aliases |}
  = {| Cached_level.names_to_types := names_to_types';
       Cached_level.aliases := aliases' |}.
Proof. intros -> ->. reflexivity. Qed.

Lemma rename_cached_level_same (cached : Cached_level.t) :
  rename_cached_level identity_renaming cached = cached.
Proof.
  destruct cached as [names_to_types aliases]. unfold rename_cached_level.
  simpl. apply cached_level_eq.
  - apply (named_bindings_same rename_binding). exact rename_binding_same.
  - apply rename_aliases_identity.
Qed.

Lemma rename_cached_level_twice (second first : renaming)
    (cached : Cached_level.t) :
  rename_cached_level (compose_renaming second first) cached
  = rename_cached_level second (rename_cached_level first cached).
Proof.
  destruct cached as [names_to_types aliases]. unfold rename_cached_level.
  simpl. apply cached_level_eq.
  - apply (named_bindings_twice rename_binding). exact rename_binding_twice.
  - apply rename_aliases_compose.
Qed.

Lemma rename_cached_level_alike (r r' : renaming) (cached : Cached_level.t) :
  (∀ x : variable,
    x ∈ cached_level_free_variables cached ->
    rename_variable r x = rename_variable r' x) ->
  rename_cached_level r cached = rename_cached_level r' cached.
Proof.
  destruct cached as [names_to_types aliases].
  unfold cached_level_free_variables, rename_cached_level. simpl.
  intros Halike. apply cached_level_eq.
  - apply (named_bindings_alike rename_binding
             (fun binding => free_variables (fst binding))).
    + exact rename_binding_alike.
    + intros x Hx. apply Halike, in_or_app. left. exact Hx.
  - apply rename_aliases_alike. intros x Hx. apply Halike, in_or_app.
    right. exact Hx.
Qed.

Lemma cached_level_free_variables_renamed (r : renaming)
    (cached : Cached_level.t) :
  cached_level_free_variables (rename_cached_level r cached)
  = map (rename_variable r) (cached_level_free_variables cached).
Proof.
  destruct cached as [names_to_types aliases].
  unfold cached_level_free_variables, rename_cached_level. simpl.
  rewrite map_app, rename_free_variables. f_equal.
  apply (named_bindings_free rename_binding
           (fun binding => free_variables (fst binding))).
  intros r' binding. exact (proj1 free_variables_rename_type r' (fst binding)).
Qed.

(* ======================================================================== *)
(* Environments                                                             *)
(* ======================================================================== *)

Lemma typing_env_eq (tenv : Typing_env.t)
    (prev_levels : list Typing_env.One_level.t)
    (current_level : Typing_env.One_level.t)
    (just_after_level : Cached_level.t) :
  prev_levels = Typing_env.prev_levels tenv ->
  current_level = Typing_env.current_level tenv ->
  just_after_level = Typing_env.just_after_level tenv ->
  {| Typing_env.defined_symbols := Typing_env.defined_symbols tenv;
     Typing_env.prev_levels := prev_levels;
     Typing_env.current_level := current_level;
     Typing_env.just_after_level := just_after_level;
     Typing_env.next_binding_time := Typing_env.next_binding_time tenv;
     Typing_env.min_binding_time := Typing_env.min_binding_time tenv;
     Typing_env.is_bottom := Typing_env.is_bottom tenv |}
  = tenv.
Proof. intros -> -> ->. destruct tenv. reflexivity. Qed.

Lemma rename_one_level_same (one_level : Typing_env.One_level.t) :
  rename_one_level identity_renaming one_level = one_level.
Proof.
  destruct one_level as [scope level]. unfold rename_one_level. simpl.
  rewrite rename_level_identity. reflexivity.
Qed.

Lemma rename_one_level_twice (second first : renaming)
    (one_level : Typing_env.One_level.t) :
  rename_one_level (compose_renaming second first) one_level
  = rename_one_level second (rename_one_level first one_level).
Proof.
  unfold rename_one_level. simpl. rewrite rename_level_compose. reflexivity.
Qed.

Lemma rename_one_level_alike (r r' : renaming)
    (one_level : Typing_env.One_level.t) :
  (∀ x : variable,
    x ∈ level_free_variables (Typing_env.One_level.level one_level) ->
    rename_variable r x = rename_variable r' x) ->
  rename_one_level r one_level = rename_one_level r' one_level.
Proof.
  intros Halike. unfold rename_one_level.
  rewrite (rename_level_extensional r r' _ Halike). reflexivity.
Qed.

Theorem rename_typing_env_identity : rename_typing_env_identity_statement.
Proof.
  intros tenv. unfold rename_typing_env. apply typing_env_eq.
  - rewrite (map_ext _ (fun one_level => one_level) rename_one_level_same).
    apply map_id.
  - apply rename_one_level_same.
  - apply rename_cached_level_same.
Qed.

Theorem rename_typing_env_compose : rename_typing_env_compose_statement.
Proof.
  intros second first tenv. unfold rename_typing_env. simpl.
  rewrite map_map, rename_one_level_twice, rename_cached_level_twice.
  rewrite (map_ext _ _ (rename_one_level_twice second first)). reflexivity.
Qed.

Theorem rename_typing_env_extensional :
  rename_typing_env_extensional_statement.
Proof.
  intros r r' tenv Halike. unfold typing_env_free_variables in Halike.
  unfold rename_typing_env.
  rewrite (rename_one_level_alike r r' (Typing_env.current_level tenv)),
    (rename_cached_level_alike r r' (Typing_env.just_after_level tenv)).
  - f_equal. apply map_ext_in. intros one_level Hin.
    apply rename_one_level_alike. intros x Hx. apply Halike, in_or_app.
    left.
    exact
      (in_flat_map_of
         (fun one_level =>
            level_free_variables (Typing_env.One_level.level one_level))
         _ one_level x Hin Hx).
  - intros x Hx. apply Halike, in_or_app. right. apply in_or_app. right.
    exact Hx.
  - intros x Hx. apply Halike, in_or_app. right. apply in_or_app. left.
    exact Hx.
Qed.

Theorem rename_typing_env_fixes : rename_typing_env_fixes_statement.
Proof.
  intros r tenv Hfixes.
  rewrite (rename_typing_env_extensional r identity_renaming tenv);
    [ apply rename_typing_env_identity | ].
  exact Hfixes.
Qed.

Theorem rename_typing_env_inverse : rename_typing_env_inverse_statement.
Proof.
  intros r tenv. rewrite <- rename_typing_env_compose.
  apply rename_typing_env_fixes. intros x _.
  rewrite rename_variable_compose. apply rename_variable_inverse_l.
Qed.

Theorem free_variables_rename_typing_env :
  free_variables_rename_typing_env_statement.
Proof.
  intros r tenv. unfold typing_env_free_variables, rename_typing_env. simpl.
  rewrite !map_app, free_variables_rename_level,
    cached_level_free_variables_renamed.
  f_equal. apply flat_map_mapped. intros one_level. simpl.
  apply free_variables_rename_level.
Qed.

(* ======================================================================== *)
(* The two instances                                                        *)
(* ======================================================================== *)

Theorem level_names_nominal : level_names_nominal_statement.
Proof.
  constructor; simpl.
  - reflexivity.
  - intros t t' Hsame. symmetry. exact Hsame.
  - intros t t' t'' -> ->. reflexivity.
  - exact rename_level_identity.
  - intros second first level.
    exact
      (rename_level_compose (variables_renaming second)
         (variables_renaming first) level).
  - intros p p' level Hsame. apply rename_level_extensional.
    intros x _. exact (Hsame x).
  - intros p t t' ->. reflexivity.
  - intros p level name. rewrite free_variables_rename_level.
    apply among_renamed_variables.
  - intros t t' -> name. reflexivity.
  - intros p level Hfixes. apply rename_level_fixes.
    intros x Hin. exact (fixes_in p _ x Hfixes Hin).
Qed.

Theorem typing_env_names_nominal : typing_env_names_nominal_statement.
Proof.
  constructor; simpl.
  - reflexivity.
  - intros t t' Hsame. symmetry. exact Hsame.
  - intros t t' t'' -> ->. reflexivity.
  - exact rename_typing_env_identity.
  - intros second first tenv.
    exact
      (rename_typing_env_compose (variables_renaming second)
         (variables_renaming first) tenv).
  - intros p p' tenv Hsame. apply rename_typing_env_extensional.
    intros x _. exact (Hsame x).
  - intros p t t' ->. reflexivity.
  - intros p tenv name. rewrite free_variables_rename_typing_env.
    apply among_renamed_variables.
  - intros t t' -> name. reflexivity.
  - intros p tenv Hfixes. apply rename_typing_env_fixes.
    intros x Hin. exact (fixes_in p _ x Hfixes Hin).
Qed.

(* ======================================================================== *)
(* Lookups and domains of renamed maps                                      *)
(* ======================================================================== *)

(* The lemmas of FlambdaEnvAliasesRenamingProofs.v on maps whose keys
   are renamed, for the two sorts of keys here: names and variables. *)
Lemma lookup_named_renamed {B C : Type} (r : renaming) (g : B -> C)
    (n : name) (bindings : Keyed.finite_map name B) :
  Keyed.lookup name_eqb (rename_name r n)
    (Keyed.map_bindings (rename_name r) g bindings)
  = option_map g (Keyed.lookup name_eqb n bindings).
Proof.
  exact (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r) g n
           bindings).
Qed.

Lemma domain_named_renamed {B C : Type} (r : renaming) (g : B -> C)
    (bindings : Keyed.finite_map name B) :
  Keyed.domain name_eqb (Keyed.map_bindings (rename_name r) g bindings)
  = map (rename_name r) (Keyed.domain name_eqb bindings).
Proof.
  exact (domain_renamed name_eqb (rename_name r) (name_eqb_renamed r) g
           bindings).
Qed.

Lemma unbind_named_renamed {B C : Type} (r : renaming) (g : B -> C)
    (n : name) (bindings : Keyed.finite_map name B) :
  Keyed.unbind name_eqb (rename_name r n)
    (Keyed.map_bindings (rename_name r) g bindings)
  = Keyed.map_bindings (rename_name r) g (Keyed.unbind name_eqb n bindings).
Proof.
  exact (unbind_renamed name_eqb (rename_name r) (name_eqb_renamed r) g n
           bindings).
Qed.

Lemma compact_named_renamed {B C : Type} (r : renaming) (g : B -> C)
    (bindings : Keyed.finite_map name B) :
  Keyed.compact name_eqb (Keyed.map_bindings (rename_name r) g bindings)
  = Keyed.map_bindings (rename_name r) g (Keyed.compact name_eqb bindings).
Proof.
  unfold Keyed.compact. rewrite domain_named_renamed.
  apply (tabulate_renamed (rename_name r) g). intros n.
  apply lookup_named_renamed.
Qed.

Lemma lookup_variable_renamed {B C : Type} (r : renaming) (g : B -> C)
    (x : variable) (bindings : finite_map B) :
  lookup (rename_variable r x)
    (Keyed.map_bindings (rename_variable r) g bindings)
  = option_map g (lookup x bindings).
Proof.
  exact (lookup_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r) g
           x bindings).
Qed.

Lemma domain_variable_renamed {B C : Type} (r : renaming) (g : B -> C)
    (bindings : finite_map B) :
  domain (Keyed.map_bindings (rename_variable r) g bindings)
  = map (rename_variable r) (domain bindings).
Proof.
  exact (domain_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r) g
           bindings).
Qed.

Lemma dedup_variables_renamed (r : renaming) (vars : list variable) :
  dedup (map (rename_variable r) vars) = map (rename_variable r) (dedup vars).
Proof.
  exact (dedup_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r)
           vars).
Qed.

Lemma inter_variables_renamed (r : renaming) (vars vars' : list variable) :
  inter (map (rename_variable r) vars) (map (rename_variable r) vars')
  = map (rename_variable r) (inter vars vars').
Proof.
  unfold inter. apply filter_renamed. intros x.
  exact
    (FlambdaEnvAliasesRenamingProofs.among_keys_renamed Pos.eqb
       (rename_variable r) (rename_variable_eqb r) x vars').
Qed.

Lemma domain_map_values_keyed {K A B : Type} (eqb : K -> K -> bool)
    (f : A -> B) (bindings : Keyed.finite_map K A) :
  Keyed.domain eqb (Keyed.map_values f bindings) = Keyed.domain eqb bindings.
Proof.
  unfold Keyed.domain, Keyed.map_values. rewrite map_map. reflexivity.
Qed.

Lemma flat_map_each_mapped {A C D : Type} (f : A -> list C)
    (f' : A -> list D) (g : C -> D) (l : list A) :
  (∀ a : A, f' a = map g (f a)) -> flat_map f' l = map g (flat_map f l).
Proof.
  intros Hf. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH, map_app. reflexivity.
Qed.

Lemma some_if_mapped {A B : Type} (c : bool) (g : A -> B) (a : A) (b : B) :
  g a = b ->
  (if c then Some b else None) = option_map g (if c then Some a else None).
Proof. intros <-. destruct c; reflexivity. Qed.

(* The variables that a level has at a binding time, if it has some. *)
Lemma consed_to_found {A B : Type} (f : A -> B) (a : A)
    (found : option (list A)) :
  match option_map (map f) found with
  | Some prev => f a :: prev
  | None => [f a]
  end
  = map f match found with
          | Some prev => a :: prev
          | None => [a]
          end.
Proof. destruct found; reflexivity. Qed.

Lemma dedup_of_found (r : renaming) (found : option (list variable)) :
  match option_map (map (rename_variable r)) found with
  | Some vars => dedup vars
  | None => []
  end
  = map (rename_variable r) match found with
                            | Some vars => dedup vars
                            | None => []
                            end.
Proof.
  destruct found as [vars | ]; simpl; [ | reflexivity ].
  apply dedup_variables_renamed.
Qed.

(* ======================================================================== *)
(* The functions of a level                                                 *)
(* ======================================================================== *)

Theorem level_empty_equivariant : level_empty_equivariant_statement.
Proof. intros r. reflexivity. Qed.

Theorem level_defined_variables_equivariant :
  level_defined_variables_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations].
  exact (domain_variable_renamed r (fun kind => kind) defined_vars).
Qed.

Theorem level_defined_names_equivariant :
  level_defined_names_equivariant_statement.
Proof.
  intros r level. unfold Typing_env_level.defined_names.
  rewrite level_defined_variables_equivariant, !map_map. reflexivity.
Qed.

Theorem level_find_kind_equivariant : level_find_kind_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations] var.
  unfold Typing_env_level.find_kind, rename_level. simpl.
  unfold Keyed.map_keys.
  rewrite (lookup_variable_renamed r (fun kind => kind) var defined_vars).
  destruct (lookup var defined_vars); reflexivity.
Qed.

Theorem level_variable_is_defined_equivariant :
  level_variable_is_defined_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations] var.
  unfold Typing_env_level.variable_is_defined, rename_level, is_bound. simpl.
  unfold Keyed.map_keys.
  rewrite (lookup_variable_renamed r (fun kind => kind) var defined_vars).
  destruct (lookup var defined_vars); reflexivity.
Qed.

Theorem level_is_empty_equivariant : level_is_empty_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations].
  unfold Typing_env_level.is_empty, rename_level. simpl.
  unfold Keyed.map_keys, Keyed.map_bindings, Keyed.map_values.
  rewrite !is_empty_map. reflexivity.
Qed.

Theorem level_add_definition_equivariant :
  level_add_definition_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations] var kind binding_time.
  unfold Typing_env_level.add_definition, rename_level. simpl.
  rewrite keyed_lookup_map_values.
  apply level_eq; [ reflexivity | | reflexivity ].
  apply f_equal2; [ | reflexivity ]. apply f_equal.
  apply (consed_to_found (rename_variable r)).
Qed.

Theorem level_add_or_replace_equation_equivariant :
  level_add_or_replace_equation_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations] n ty.
  unfold Typing_env_level.add_or_replace_equation, rename_level. simpl.
  rewrite is_obviously_unknown_renamed.
  destruct (is_obviously_unknown ty); [ | reflexivity ].
  apply level_eq; try reflexivity. apply unbind_named_renamed.
Qed.

Theorem level_concat_equivariant : level_concat_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations]
    [defined_vars' binding_times' equations'].
  unfold Typing_env_level.concat, rename_level. simpl.
  unfold Keyed.map_keys.
  rewrite !domain_variable_renamed, inter_variables_renamed, is_empty_map,
    !domain_map_values_keyed.
  apply (some_if_mapped _ (rename_level r)). unfold rename_level. simpl.
  unfold Keyed.map_keys, Keyed.map_bindings, Keyed.map_values.
  rewrite !map_app. reflexivity.
Qed.

Theorem level_as_extension_equivariant :
  level_as_extension_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations].
  unfold Typing_env_level.as_extension_without_bindings, rename_level. simpl.
  f_equal. exact (compact_named_renamed r (rename_type r) equations).
Qed.

Theorem level_variables_in_order_equivariant :
  level_variables_in_order_equivariant_statement.
Proof.
  intros r [defined_vars binding_times equations].
  unfold Typing_env_level.variables_in_order, rename_level. simpl.
  rewrite domain_map_values_keyed. apply flat_map_each_mapped.
  intros binding_time. rewrite keyed_lookup_map_values.
  apply (dedup_of_found r).
Qed.

Theorem level_fold_on_defined_vars_equivariant :
  level_fold_on_defined_vars_equivariant_statement.
Proof.
  intros A r rename_acc f f' level init Hf.
  unfold Typing_env_level.fold_on_defined_vars.
  rewrite level_variables_in_order_equivariant.
  change (Some (rename_acc init)) with (option_map rename_acc (Some init)).
  generalize (Some init) as acc.
  induction (Typing_env_level.variables_in_order level) as [ | var vars IH];
    intros acc; simpl; [ reflexivity | ].
  rewrite <- IH. f_equal. destruct acc as [acc | ]; simpl; [ | reflexivity ].
  rewrite level_find_kind_equivariant.
  destruct (Typing_env_level.find_kind level var) as [kind | ]; simpl;
    [ | reflexivity ].
  rewrite Hf. reflexivity.
Qed.

(* ======================================================================== *)
(* What the environment knows of its names                                  *)
(* ======================================================================== *)

Theorem cached_level_empty_equivariant :
  cached_level_empty_equivariant_statement.
Proof. intros r. reflexivity. Qed.

Theorem cached_level_add_or_replace_binding_equivariant :
  cached_level_add_or_replace_binding_equivariant_statement.
Proof. intros r cached n ty binding_time name_mode. reflexivity. Qed.

Theorem cached_level_replace_variable_binding_equivariant :
  cached_level_replace_variable_binding_equivariant_statement.
Proof.
  intros r [names_to_types aliases] var ty.
  unfold Cached_level.replace_variable_binding, rename_cached_level. simpl.
  change (Name_var (rename_variable r var))
    with (rename_name r (Name_var var)).
  rewrite lookup_named_renamed.
  destruct (Keyed.lookup name_eqb (Name_var var) names_to_types)
    as [[ty' binding_time] | ];
    reflexivity.
Qed.

Theorem cached_level_with_aliases_equivariant :
  cached_level_with_aliases_equivariant_statement.
Proof. intros r cached aliases. reflexivity. Qed.

(* ======================================================================== *)
(* What an environment is asked                                             *)
(* ======================================================================== *)

Ltac open_binding :=
  cbn [option_map fst snd]; unfold rename_binding; cbn [fst snd].

Lemma names_to_types_renamed (r : renaming) (tenv : Typing_env.t) :
  Typing_env.names_to_types (rename_typing_env r tenv)
  = Keyed.map_bindings (rename_name r) (rename_binding r)
      (Typing_env.names_to_types tenv).
Proof. reflexivity. Qed.

Theorem create_equivariant : create_equivariant_statement.
Proof. intros r. reflexivity. Qed.

Theorem aliases_equivariant : aliases_equivariant_statement.
Proof. intros r tenv. reflexivity. Qed.

Theorem names_to_types_equivariant : names_to_types_equivariant_statement.
Proof. exact names_to_types_renamed. Qed.

Theorem current_scope_equivariant : current_scope_equivariant_statement.
Proof. intros r tenv. reflexivity. Qed.

Theorem name_domain_equivariant : name_domain_equivariant_statement.
Proof.
  intros r tenv. unfold Typing_env.name_domain.
  rewrite names_to_types_renamed, domain_named_renamed, map_app, map_map.
  reflexivity.
Qed.

Lemma check_optional_kind_matches_renamed (r : renaming) (ty : flambda_type)
    (kind : option flambda_kind) :
  Typing_env.check_optional_kind_matches (rename_type r ty) kind
  = Typing_env.check_optional_kind_matches ty kind.
Proof.
  destruct kind as [kind | ]; simpl; [ | reflexivity ].
  rewrite kind_rename_type. reflexivity.
Qed.

Theorem find_unscoped_equivariant : find_unscoped_equivariant_statement.
Proof.
  intros r tenv n kind. unfold Typing_env.find_with_binding_time_and_mode'.
  rewrite names_to_types_renamed, lookup_named_renamed.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty binding_time] | ];
    open_binding.
  - rewrite check_optional_kind_matches_renamed.
    destruct (Typing_env.check_optional_kind_matches ty kind);
      [ | reflexivity ].
    change (Typing_env.is_bottom (rename_typing_env r tenv))
      with (Typing_env.is_bottom tenv).
    destruct (Typing_env.is_bottom tenv); open_binding;
      rewrite ?kind_rename_type, ?rename_type_bottom; reflexivity.
  - destruct n as [x | sym]; [ reflexivity | ].
    cbn [rename_name].
    change (Typing_env.defined_symbols (rename_typing_env r tenv))
      with (Typing_env.defined_symbols tenv).
    destruct
      (among sym (Typing_env.defined_symbols tenv)
       && Typing_env.check_optional_kind_matches
            (fst Typing_env.initial_symbol_type) kind);
      reflexivity.
Qed.

Theorem find_equivariant : find_equivariant_statement.
Proof.
  intros r tenv n kind. unfold Typing_env.find.
  rewrite find_unscoped_equivariant.
  destruct (Typing_env.find_with_binding_time_and_mode' tenv n kind)
    as [[ty binding_time] | ];
    reflexivity.
Qed.

Theorem find_with_binding_time_and_mode_equivariant :
  find_with_binding_time_and_mode_equivariant_statement.
Proof.
  intros r tenv n kind. unfold Typing_env.find_with_binding_time_and_mode.
  rewrite find_unscoped_equivariant.
  destruct (Typing_env.find_with_binding_time_and_mode' tenv n kind)
    as [[ty binding_time] | ];
    reflexivity.
Qed.

Theorem binding_time_and_mode_equivariant :
  binding_time_and_mode_equivariant_statement.
Proof.
  intros r tenv [x | sym]; [ | reflexivity ].
  pose proof
    (find_with_binding_time_and_mode_equivariant r tenv (Name_var x) None)
    as Hfind.
  unfold Typing_env.binding_time_and_mode. cbn [rename_name] in *.
  rewrite Hfind.
  destruct (Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None)
    as [[ty binding_time] | ];
    reflexivity.
Qed.

Theorem binding_time_and_mode_of_simple_equivariant :
  binding_time_and_mode_of_simple_equivariant_statement.
Proof.
  intros r tenv [x | sym | c]; [ | reflexivity | reflexivity ].
  exact (binding_time_and_mode_equivariant r tenv (Name_var x)).
Qed.

Theorem binding_time_resolver_equivariant :
  binding_time_resolver_equivariant_statement.
Proof.
  intros r tenv n. unfold Typing_env.binding_time_resolver.
  rewrite names_to_types_renamed, lookup_named_renamed.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty binding_time] | ];
    [ reflexivity | destruct n; reflexivity ].
Qed.

Theorem mem_equivariant : mem_equivariant_statement.
Proof.
  intros r min_name_mode tenv [x | sym]; [ | reflexivity ].
  unfold Typing_env.mem.
  rewrite names_to_types_renamed, lookup_named_renamed.
  destruct
    (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [[ty binding_time] | ];
    reflexivity.
Qed.

Theorem mem_simple_equivariant : mem_simple_equivariant_statement.
Proof.
  intros r min_name_mode tenv [x | sym | c]; [ | reflexivity | reflexivity ].
  exact (mem_equivariant r min_name_mode tenv (Name_var x)).
Qed.

Theorem alias_is_bound_strictly_earlier_equivariant :
  alias_is_bound_strictly_earlier_equivariant_statement.
Proof.
  intros r tenv bound_name alias.
  unfold Typing_env.alias_is_bound_strictly_earlier.
  rewrite binding_time_and_mode_equivariant,
    binding_time_and_mode_of_simple_equivariant.
  reflexivity.
Qed.

(* What the functions of the aliases take of the renamed environment is
   what the renaming of FlambdaEnvAliasesRenamingSpec.v makes of what
   they take of the environment, as far as they read it: they read the
   resolver, and of the map from names to types the binding times and
   the name modes only (FlambdaEnvAliasesSpec.v,
   [get_canonical_element_exn_alike_statement] and the statements
   beside it). *)
Lemma resolver_of_renamed (r : renaming) (tenv : Typing_env.t) (n : name) :
  Typing_env.binding_time_resolver (rename_typing_env r tenv) n
  = rename_resolver r (Typing_env.binding_time_resolver tenv) n.
Proof.
  unfold rename_resolver.
  rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n) at 1.
  apply binding_time_resolver_equivariant.
Qed.

Lemma times_of_renamed (r : renaming) (tenv : Typing_env.t) (n : name) :
  option_map snd
    (Keyed.lookup name_eqb n
       (Typing_env.names_to_types (rename_typing_env r tenv)))
  = option_map snd
      (Keyed.lookup name_eqb n
         (rename_binding_times r (Typing_env.names_to_types tenv))).
Proof.
  rewrite names_to_types_renamed.
  unfold rename_binding_times, Keyed.map_bindings.
  induction (Typing_env.names_to_types tenv) as [ | [n' [ty bt]] bindings IH];
    simpl; [ reflexivity | ].
  destruct n' as [x | sym]; simpl;
    match goal with
    | |- context [name_eqb n ?n'] => destruct (name_eqb n n')
    end;
    first [ reflexivity | exact IH ].
Qed.

Lemma canonical_element_of_renamed (r : renaming) (tenv : Typing_env.t)
    (simple : simple) (elt_name_mode min_name_mode : Name_mode.t) :
  Aliases.get_canonical_element_exn
    (Typing_env.binding_time_resolver (rename_typing_env r tenv))
    (Typing_env.names_to_types (rename_typing_env r tenv))
    (Typing_env.aliases (rename_typing_env r tenv)) (rename_simple r simple)
    elt_name_mode min_name_mode
    (Typing_env.min_binding_time (rename_typing_env r tenv))
  = option_map (rename_simple r)
      (Aliases.get_canonical_element_exn
         (Typing_env.binding_time_resolver tenv)
         (Typing_env.names_to_types tenv) (Typing_env.aliases tenv) simple
         elt_name_mode min_name_mode (Typing_env.min_binding_time tenv)).
Proof.
  rewrite
    (get_canonical_element_exn_alike_here _
       (rename_resolver r (Typing_env.binding_time_resolver tenv)) _
       (rename_binding_times r (Typing_env.names_to_types tenv))
       (resolver_of_renamed r tenv) (times_of_renamed r tenv)).
  exact
    (rename_canonical_at_mode flambda_type r
       (Typing_env.binding_time_resolver tenv)
       (Typing_env.names_to_types tenv) (Typing_env.aliases tenv) simple
       elt_name_mode min_name_mode (Typing_env.min_binding_time tenv)).
Qed.

Theorem get_canonical_ignoring_name_mode_equivariant :
  get_canonical_ignoring_name_mode_equivariant_statement.
Proof.
  intros r tenv [x | sym | c]; [ | | reflexivity ].
  - exact (rename_canonical r (Typing_env.aliases tenv) (Name_var x)).
  - exact (rename_canonical r (Typing_env.aliases tenv) (Name_symbol sym)).
Qed.

Theorem get_canonical_simple_exn_equivariant :
  get_canonical_simple_exn_equivariant_statement.
Proof.
  intros r tenv min_name_mode name_mode_of_existing_simple simple.
  unfold Typing_env.get_canonical_simple_exn.
  rewrite binding_time_and_mode_of_simple_equivariant.
  destruct
    match name_mode_of_existing_simple with
    | Some name_mode => Some name_mode
    | None =>
      option_map Binding_time.With_name_mode.name_mode
        (Typing_env.binding_time_and_mode_of_simple tenv simple)
    end
    as [name_mode_simple | ];
    [ | reflexivity ].
  cbv beta iota. rewrite canonical_element_of_renamed. reflexivity.
Qed.

Theorem get_alias_then_canonical_simple_exn_equivariant :
  get_alias_then_canonical_simple_exn_equivariant_statement.
Proof.
  intros r tenv min_name_mode name_mode_of_existing_simple typ.
  unfold Typing_env.get_alias_then_canonical_simple_exn.
  rewrite get_alias_opt_renamed.
  destruct (get_alias_opt typ) as [simple | ]; [ | reflexivity ].
  apply get_canonical_simple_exn_equivariant.
Qed.

(* The type that a [Simple] has in a term, before its canonical element
   is asked for. *)
Lemma typed_simple_renamed (r : renaming) (tenv : Typing_env.t)
    (simple : simple) :
  match rename_simple r simple with
  | Const const =>
    Some (type_for_const const, Binding_time.With_name_mode.consts)
  | Var x =>
    Typing_env.find_with_binding_time_and_mode (rename_typing_env r tenv)
      (Name_var x) None
  | Symbol s =>
    Typing_env.find_with_binding_time_and_mode (rename_typing_env r tenv)
      (Name_symbol s) None
  end
  = option_map (rename_binding r)
      match simple with
      | Const const =>
        Some (type_for_const const, Binding_time.With_name_mode.consts)
      | Var x =>
        Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None
      | Symbol s =>
        Typing_env.find_with_binding_time_and_mode tenv (Name_symbol s) None
      end.
Proof.
  destruct simple as [x | sym | c]; cbn [rename_simple].
  - exact
      (find_with_binding_time_and_mode_equivariant r tenv (Name_var x) None).
  - exact
      (find_with_binding_time_and_mode_equivariant r tenv (Name_symbol sym)
         None).
  - open_binding.
    rewrite rename_type_type_for_const. reflexivity.
Qed.

Theorem type_simple_in_term_exn_equivariant :
  type_simple_in_term_exn_equivariant_statement.
Proof.
  intros r tenv min_name_mode simple.
  unfold Typing_env.type_simple_in_term_exn.
  rewrite typed_simple_renamed.
  destruct
    match simple with
    | Const const =>
      Some (type_for_const const, Binding_time.With_name_mode.consts)
    | Var x =>
      Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None
    | Symbol s =>
      Typing_env.find_with_binding_time_and_mode tenv (Name_symbol s) None
    end
    as [[ty binding_time] | ];
    [ | reflexivity ].
  cbv beta iota; open_binding.
  rewrite canonical_element_of_renamed, kind_rename_type.
  destruct
    (Aliases.get_canonical_element_exn _ _ _ simple _ _ _)
    as [alias | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite rename_type_alias_type_of. reflexivity.
Qed.

(* ======================================================================== *)
(* What is done to an environment                                           *)
(* ======================================================================== *)

Theorem make_bottom_equivariant : make_bottom_equivariant_statement.
Proof. intros r tenv. reflexivity. Qed.

Theorem increment_scope_equivariant : increment_scope_equivariant_statement.
Proof. intros r tenv. reflexivity. Qed.

Theorem bump_current_level_scope_equivariant :
  bump_current_level_scope_equivariant_statement.
Proof. intros r tenv. reflexivity. Qed.

Theorem closure_env_equivariant : closure_env_equivariant_statement.
Proof. intros r tenv. reflexivity. Qed.

Theorem with_aliases_equivariant : with_aliases_equivariant_statement.
Proof. intros r tenv aliases. reflexivity. Qed.

Lemma with_current_level_renamed (r : renaming) (tenv : Typing_env.t)
    (current_level : Typing_env.One_level.t)
    (just_after_level : Cached_level.t) :
  Typing_env.with_current_level (rename_typing_env r tenv)
    (rename_one_level r current_level)
    (rename_cached_level r just_after_level)
  = rename_typing_env r
      (Typing_env.with_current_level tenv current_level just_after_level).
Proof. reflexivity. Qed.

Lemma with_current_level_and_next_binding_time_renamed (r : renaming)
    (tenv : Typing_env.t) (current_level : Typing_env.One_level.t)
    (just_after_level : Cached_level.t) (next_binding_time : Binding_time.t) :
  Typing_env.with_current_level_and_next_binding_time
    (rename_typing_env r tenv) (rename_one_level r current_level)
    (rename_cached_level r just_after_level) next_binding_time
  = rename_typing_env r
      (Typing_env.with_current_level_and_next_binding_time tenv current_level
         just_after_level next_binding_time).
Proof. reflexivity. Qed.

Theorem add_variable_definition_equivariant :
  add_variable_definition_equivariant_statement.
Proof.
  intros r tenv var kind name_mode. unfold Typing_env.add_variable_definition.
  cbv zeta. rewrite <- with_current_level_and_next_binding_time_renamed.
  f_equal.
  - exact
      (f_equal (Typing_env.One_level.create (Typing_env.current_scope tenv))
         (level_add_definition_equivariant r _ var kind _)).
  - rewrite <- cached_level_add_or_replace_binding_equivariant,
      rename_type_unknown.
    reflexivity.
Qed.

Theorem add_symbol_definition_equivariant :
  add_symbol_definition_equivariant_statement.
Proof. intros r tenv sym. reflexivity. Qed.

Theorem add_symbol_definitions_equivariant :
  add_symbol_definitions_equivariant_statement.
Proof. intros r tenv syms. reflexivity. Qed.

Theorem add_definition_equivariant : add_definition_equivariant_statement.
Proof.
  intros r tenv [var | sym] name_mode kind;
    cbn [rename_name Typing_env.add_definition option_map].
  - rewrite add_variable_definition_equivariant. reflexivity.
  - destruct (Name_mode.equal name_mode Name_mode.normal); reflexivity.
Qed.

Theorem add_definitions_of_params_equivariant :
  add_definitions_of_params_equivariant_statement.
Proof.
  intros r tenv params. unfold Typing_env.add_definitions_of_params.
  revert tenv. induction params as [ | [var kind] params IH]; intros tenv;
    simpl; [ reflexivity | ].
  rewrite add_variable_definition_equivariant. apply IH.
Qed.

Theorem replace_equation_equivariant : replace_equation_equivariant_statement.
Proof.
  intros r tenv n ty. unfold Typing_env.replace_equation. cbv zeta.
  rewrite <- with_current_level_renamed. f_equal.
  - exact
      (f_equal (Typing_env.One_level.create (Typing_env.current_scope tenv))
         (level_add_or_replace_equation_equivariant r _ n ty)).
  - destruct n as [var | sym]; [ | reflexivity ].
    exact
      (cached_level_replace_variable_binding_equivariant r
         (Typing_env.just_after_level tenv) var ty).
Qed.

Theorem aliases_add_equivariant : aliases_add_equivariant_statement.
Proof.
  intros r tenv canonical_element1 canonical_element2.
  unfold Typing_env.aliases_add.
  rewrite
    (add_alike_here _
       (rename_resolver r (Typing_env.binding_time_resolver tenv)) _
       (rename_binding_times r (Typing_env.names_to_types tenv))
       (resolver_of_renamed r tenv) (times_of_renamed r tenv)).
  change (Typing_env.aliases (rename_typing_env r tenv))
    with (rename_aliases r (Typing_env.aliases tenv)).
  rewrite (rename_add flambda_type).
  destruct
    (Aliases.add (Typing_env.binding_time_resolver tenv)
       (Typing_env.names_to_types tenv) (Typing_env.aliases tenv)
       canonical_element1 canonical_element2)
    as [[result | ] | ];
    reflexivity.
Qed.

Theorem add_alias_equivariant : add_alias_equivariant_statement.
Proof.
  intros r tenv canonical_element1 canonical_element2.
  unfold Typing_env.add_alias. rewrite aliases_add_equivariant.
  destruct (Typing_env.aliases_add tenv canonical_element1 canonical_element2)
    as [[result | ] | ];
    reflexivity.
Qed.

Lemma cut_loop_renamed (r : renaming) (cut_after : Scope.t)
    (levels : list Typing_env.One_level.t) :
  ∀ result : Typing_env_level.t,
    Typing_env.cut_loop cut_after (rename_level r result)
      (map (rename_one_level r) levels)
    = option_map (rename_level r) (Typing_env.cut_loop cut_after result levels).
Proof.
  induction levels as [ | one_level levels IH]; intros result;
    cbn [map Typing_env.cut_loop]; [ reflexivity | ].
  change (Typing_env.One_level.scope (rename_one_level r one_level))
    with (Typing_env.One_level.scope one_level).
  destruct (cut_after <? Typing_env.One_level.scope one_level)%Z;
    [ | reflexivity ].
  change (Typing_env.One_level.level (rename_one_level r one_level))
    with (rename_level r (Typing_env.One_level.level one_level)).
  rewrite level_concat_equivariant.
  destruct
    (Typing_env_level.concat (Typing_env.One_level.level one_level) result)
    as [result' | ];
    [ apply IH | reflexivity ].
Qed.

Theorem cut_equivariant : cut_equivariant_statement.
Proof.
  intros r tenv cut_after. unfold Typing_env.cut.
  change (Typing_env.current_scope (rename_typing_env r tenv))
    with (Typing_env.current_scope tenv).
  destruct (Typing_env.current_scope tenv <=? cut_after)%Z; [ reflexivity | ].
  exact
    (cut_loop_renamed r cut_after (Typing_env.prev_levels tenv)
       (Typing_env.One_level.level (Typing_env.current_level tenv))).
Qed.

Theorem cut_as_extension_equivariant : cut_as_extension_equivariant_statement.
Proof.
  intros r tenv cut_after. unfold Typing_env.cut_as_extension.
  rewrite cut_equivariant.
  destruct (Typing_env.cut tenv cut_after) as [level | ]; [ | reflexivity ].
  cbn [option_map]. rewrite level_as_extension_equivariant. reflexivity.
Qed.

(* ======================================================================== *)
(* What satisfies a renamed environment                                     *)
(* ======================================================================== *)

(* Each sense of satisfaction is kept by a renaming. The converse is
   the same fact of the inverse renaming, which takes the renamed
   valuation and the renamed environment back. *)

Lemma name_renamed_back (r : renaming) (n : name) :
  rename_name (inverse_renaming r) (rename_name r n) = n.
Proof.
  destruct n as [x | sym]; simpl; [ | reflexivity ].
  rewrite rename_variable_inverse_l. reflexivity.
Qed.

Lemma value_of_renamed_name (r : renaming) (valuation : env) (n : name) :
  value_of_name (rename_env r valuation) (rename_name r n)
  = value_of_name valuation n.
Proof. exact (value_of_name_rename_env r valuation n). Qed.

Lemma holds_poison_renamed (r : renaming) (valuation : env) (n : name) :
  holds_poison valuation n ->
  holds_poison (rename_env r valuation) (rename_name r n).
Proof.
  intros (v & Hvalue & Hpoison). exists v.
  rewrite value_of_renamed_name. split; assumption.
Qed.

Lemma relation_on_poison_renamed (r : renaming) (valuation : env)
    (ty : flambda_type) :
  relation_on_poison valuation ty ->
  relation_on_poison (rename_env r valuation) (rename_type r ty).
Proof.
  intros (head & relation & scrutinee & -> & Hin & Hrelation & Hpoison).
  exists (rename_head_of_kind_naked_immediate r head), relation,
    (rename_name r scrutinee).
  split; [ reflexivity | ]. split; [ | split; [ exact Hrelation | ] ].
  - exact
      (in_map (fun relation => (fst relation, rename_name r (snd relation)))
         _ _ Hin).
  - exact (holds_poison_renamed r _ _ Hpoison).
Qed.

Lemma exempt_by_poison_renamed (r : renaming) (valuation : env)
    (tenv : Typing_env.t) (n : name) :
  exempt_by_poison valuation tenv n ->
  exempt_by_poison (rename_env r valuation) (rename_typing_env r tenv)
    (rename_name r n).
Proof.
  intros [Hpoison | (ty & binding_time_and_mode & Hlookup & Hrelation)].
  - left. exact (holds_poison_renamed r _ _ Hpoison).
  - right. exists (rename_type r ty), binding_time_and_mode. split.
    + rewrite names_to_types_renamed, lookup_named_renamed, Hlookup.
      reflexivity.
    + exact (relation_on_poison_renamed r _ _ Hrelation).
Qed.

Theorem exempt_by_poison_rename : exempt_by_poison_rename_statement.
Proof.
  intros r valuation tenv n. split; [ | apply exempt_by_poison_renamed ].
  intros H.
  pose proof (exempt_by_poison_renamed (inverse_renaming r) _ _ _ H) as H'.
  rewrite rename_env_inverse_l, rename_typing_env_inverse, name_renamed_back
    in H'.
  exact H'.
Qed.

Lemma has_type_unless_renamed (r : renaming) (exempt exempt' : Prop)
    (statics : static_env) (valuation : env) (v : value) (ty : flambda_type) :
  (exempt -> exempt') ->
  has_type_unless exempt statics valuation v ty ->
  has_type_unless exempt' statics (rename_env r valuation) v
    (rename_type r ty).
Proof.
  intros Himplies [[Hexempt Hkind] | Hsatisfied].
  - left. split; [ exact (Himplies Hexempt) | ].
    rewrite kind_rename_type. exact Hkind.
  - right. apply type_satisfied_rename. exact Hsatisfied.
Qed.

Lemma names_have_their_types_but_renamed (r : renaming)
    (exempt exempt' : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) :
  (∀ n : name, exempt n -> exempt' (rename_name r n)) ->
  names_have_their_types_but exempt statics valuation tenv ->
  names_have_their_types_but exempt' statics (rename_env r valuation)
    (rename_typing_env r tenv).
Proof.
  intros Himplies H n ty binding_time_and_mode v.
  rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n).
  set (n0 := rename_name (inverse_renaming r) n).
  rewrite names_to_types_renamed, lookup_named_renamed,
    value_of_renamed_name.
  destruct (Keyed.lookup name_eqb n0 (Typing_env.names_to_types tenv))
    as [[ty0 binding_time_and_mode0] | ] eqn:Hlookup; [ | discriminate ].
  simpl. unfold rename_binding. simpl. intros Hrenamed Hvalue.
  injection Hrenamed as <- <-.
  apply (has_type_unless_renamed r (exempt n0)); [ apply Himplies | ].
  exact (H n0 ty0 binding_time_and_mode0 v Hlookup Hvalue).
Qed.

Lemma satisfied_but_renamed (r : renaming) (exempt exempt' : name -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t) :
  (∀ n : name, exempt n -> exempt' (rename_name r n)) ->
  typing_env_satisfied_but exempt statics valuation tenv ->
  typing_env_satisfied_but exempt' statics (rename_env r valuation)
    (rename_typing_env r tenv).
Proof.
  intros Himplies (Hbottom & Htypes & Haliases).
  split; [ exact Hbottom | ]. split.
  - exact (names_have_their_types_but_renamed r _ _ _ _ _ Himplies Htypes).
  - change (Typing_env.aliases (rename_typing_env r tenv))
      with (rename_aliases r (Typing_env.aliases tenv)).
    apply rename_respected. exact Haliases.
Qed.

(* The converse of each, from the inverse renaming. *)
Lemma satisfied_but_renamed_back (r : renaming) (exempt exempt' : name -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t) :
  (∀ n : name, exempt' n -> exempt (rename_name (inverse_renaming r) n)) ->
  typing_env_satisfied_but exempt' statics (rename_env r valuation)
    (rename_typing_env r tenv) ->
  typing_env_satisfied_but exempt statics valuation tenv.
Proof.
  intros Himplies H.
  pose proof
    (satisfied_but_renamed (inverse_renaming r) _ _ _ _ _ Himplies H) as H'.
  rewrite rename_env_inverse_l, rename_typing_env_inverse in H'. exact H'.
Qed.

Theorem typing_env_satisfied_but_rename :
  typing_env_satisfied_but_rename_statement.
Proof.
  intros r exempt statics valuation tenv. split.
  - apply satisfied_but_renamed_back. intros n Hexempt. exact Hexempt.
  - apply satisfied_but_renamed. intros n Hexempt. unfold rename_exempt.
    rewrite name_renamed_back. exact Hexempt.
Qed.

Theorem typing_env_satisfied_strictly_rename :
  typing_env_satisfied_strictly_rename_statement.
Proof.
  intros r statics valuation tenv. unfold typing_env_satisfied_strictly. split.
  - apply satisfied_but_renamed_back. intros n [].
  - apply satisfied_but_renamed. intros n [].
Qed.

Theorem typing_env_satisfied_rename : typing_env_satisfied_rename_statement.
Proof.
  intros r statics valuation tenv. unfold typing_env_satisfied. split.
  - apply satisfied_but_renamed_back. intros n Hexempt.
    apply (exempt_by_poison_rename r).
    rewrite FlambdaTypeRenameProofs.rename_name_inverse_r. exact Hexempt.
  - apply satisfied_but_renamed. intros n.
    apply exempt_by_poison_renamed.
Qed.

Theorem in_scope_in_terms_rename : in_scope_in_terms_rename_statement.
Proof.
  intros r tenv x. unfold in_scope_in_terms.
  change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
  rewrite mem_equivariant. reflexivity.
Qed.

Lemma bound_in_terms_renamed (r : renaming) (tenv : Typing_env.t)
    (vars : env) :
  bound_in_terms tenv vars ->
  bound_in_terms (rename_typing_env r tenv) (rename_env r vars).
Proof.
  intros H x. rewrite <- (rename_variable_inverse_r r x). intros Hin.
  apply (proj1 (in_scope_in_terms_rename r tenv _)) in Hin.
  rewrite lookup_rename_env. exact (H _ Hin).
Qed.

Lemma agrees_in_terms_renamed (r : renaming) (tenv : Typing_env.t)
    (valuation vars : env) :
  agrees_in_terms tenv valuation vars ->
  agrees_in_terms (rename_typing_env r tenv) (rename_env r valuation)
    (rename_env r vars).
Proof.
  intros H x. rewrite <- (rename_variable_inverse_r r x). intros Hin.
  apply (proj1 (in_scope_in_terms_rename r tenv _)) in Hin.
  rewrite !lookup_rename_env. exact (H _ Hin).
Qed.

Lemma env_satisfies_renamed (r : renaming) (tenv : Typing_env.t)
    (statics : static_env) (vars : env) :
  env_satisfies tenv statics vars ->
  env_satisfies (rename_typing_env r tenv) statics (rename_env r vars).
Proof.
  intros (Hbound & valuation & Hsatisfied & Hagrees).
  split; [ exact (bound_in_terms_renamed r _ _ Hbound) | ].
  exists (rename_env r valuation). split.
  - apply typing_env_satisfied_rename. exact Hsatisfied.
  - exact (agrees_in_terms_renamed r _ _ _ Hagrees).
Qed.

Theorem env_satisfies_rename : env_satisfies_rename_statement.
Proof.
  intros r tenv statics vars. split; [ | apply env_satisfies_renamed ].
  intros H.
  pose proof (env_satisfies_renamed (inverse_renaming r) _ _ _ H) as H'.
  rewrite rename_env_inverse_l, rename_typing_env_inverse in H'. exact H'.
Qed.

Lemma env_satisfies_strictly_renamed (r : renaming) (tenv : Typing_env.t)
    (statics : static_env) (vars : env) :
  env_satisfies_strictly tenv statics vars ->
  env_satisfies_strictly (rename_typing_env r tenv) statics
    (rename_env r vars).
Proof.
  intros (Hbound & valuation & Hsatisfied & Hagrees).
  split; [ exact (bound_in_terms_renamed r _ _ Hbound) | ].
  exists (rename_env r valuation). split.
  - apply typing_env_satisfied_strictly_rename. exact Hsatisfied.
  - exact (agrees_in_terms_renamed r _ _ _ Hagrees).
Qed.

Theorem env_satisfies_strictly_rename :
  env_satisfies_strictly_rename_statement.
Proof.
  intros r tenv statics vars.
  split; [ | apply env_satisfies_strictly_renamed ].
  intros H.
  pose proof (env_satisfies_strictly_renamed (inverse_renaming r) _ _ _ H)
    as H'.
  rewrite rename_env_inverse_l, rename_typing_env_inverse in H'. exact H'.
Qed.

(* ======================================================================== *)
(* The invariant of a renamed environment                                   *)
(* ======================================================================== *)

(* Each condition of the invariant is kept by a renaming. The converse
   is the same fact of the inverse renaming. *)

Lemma lookup_of_renamed (r : renaming) (tenv : Typing_env.t) (n : name)
    (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
  = Some (ty, binding_time_and_mode) ->
  Keyed.lookup name_eqb (rename_name r n)
    (Typing_env.names_to_types (rename_typing_env r tenv))
  = Some (rename_type r ty, binding_time_and_mode).
Proof.
  intros Hlookup.
  rewrite names_to_types_renamed, lookup_named_renamed, Hlookup. reflexivity.
Qed.

Lemma lookup_in_renamed (r : renaming) (tenv : Typing_env.t) (n : name)
    (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n
    (Typing_env.names_to_types (rename_typing_env r tenv))
  = Some (ty, binding_time_and_mode) ->
  ∃ ty0 : flambda_type,
    ty = rename_type r ty0
    ∧ Keyed.lookup name_eqb (rename_name (inverse_renaming r) n)
        (Typing_env.names_to_types tenv)
      = Some (ty0, binding_time_and_mode).
Proof.
  rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n) at 1.
  rewrite names_to_types_renamed, lookup_named_renamed.
  destruct
    (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n)
       (Typing_env.names_to_types tenv))
    as [[ty0 binding_time_and_mode0] | ];
    [ | discriminate ].
  unfold rename_binding. simpl. intros Hrenamed.
  injection Hrenamed as <- <-. exists ty0. split; reflexivity.
Qed.

Lemma is_bound_named_renamed {B C : Type} (r : renaming) (g : B -> C)
    (n : name) (bindings : Keyed.finite_map name B) :
  Keyed.is_bound name_eqb (rename_name r n)
    (Keyed.map_bindings (rename_name r) g bindings)
  = Keyed.is_bound name_eqb n bindings.
Proof.
  unfold Keyed.is_bound. rewrite lookup_named_renamed.
  destruct (Keyed.lookup name_eqb n bindings); reflexivity.
Qed.

Lemma simple_of_renamed_name (r : renaming) (n : name) :
  simple_of_name (rename_name r n) = rename_simple r (simple_of_name n).
Proof. destruct n; reflexivity. Qed.

(* The names and their types. *)

Lemma binding_times_in_range_renamed (r : renaming) (tenv : Typing_env.t) :
  binding_times_in_range tenv ->
  binding_times_in_range (rename_typing_env r tenv).
Proof.
  intros H x ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & _ & Hlookup).
  exact (H _ _ _ Hlookup).
Qed.

Lemma binding_times_distinct_renamed (r : renaming) (tenv : Typing_env.t) :
  binding_times_distinct tenv ->
  binding_times_distinct (rename_typing_env r tenv).
Proof.
  intros H x y ty ty' binding_time_and_mode binding_time_and_mode' Hx Hy
    Hsame.
  apply lookup_in_renamed in Hx as (ty0 & _ & Hx).
  apply lookup_in_renamed in Hy as (ty0' & _ & Hy).
  pose proof (H _ _ _ _ _ _ Hx Hy Hsame) as Heq.
  apply (f_equal (rename_variable r)) in Heq.
  rewrite !rename_variable_inverse_r in Heq. exact Heq.
Qed.

Lemma types_closed_renamed (r : renaming) (tenv : Typing_env.t) :
  types_closed tenv -> types_closed (rename_typing_env r tenv).
Proof.
  intros H n ty binding_time_and_mode x Hlookup Hin.
  apply lookup_in_renamed in Hlookup as (ty0 & -> & Hlookup).
  rewrite (proj1 free_variables_rename_type) in Hin.
  apply in_map_iff in Hin as (x0 & <- & Hin).
  rewrite names_to_types_renamed.
  change (Name_var (rename_variable r x0)) with (rename_name r (Name_var x0)).
  rewrite is_bound_named_renamed. exact (H _ _ _ _ Hlookup Hin).
Qed.

Lemma symbols_defined_renamed (r : renaming) (tenv : Typing_env.t) :
  symbols_defined tenv -> symbols_defined (rename_typing_env r tenv).
Proof.
  intros H s ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & _ & Hlookup).
  exact (H _ _ _ Hlookup).
Qed.

Lemma symbols_recorded_renamed (r : renaming) (tenv : Typing_env.t) :
  symbols_recorded tenv -> symbols_recorded (rename_typing_env r tenv).
Proof.
  intros H s ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & _ & Hlookup).
  exact (H _ _ _ Hlookup).
Qed.

(* The levels. *)

Lemma all_levels_renamed (r : renaming) (tenv : Typing_env.t) :
  all_levels (rename_typing_env r tenv)
  = map (rename_level r) (all_levels tenv).
Proof. unfold all_levels. simpl. rewrite !map_map. reflexivity. Qed.

Lemma all_scopes_renamed (r : renaming) (tenv : Typing_env.t) :
  all_scopes (rename_typing_env r tenv) = all_scopes tenv.
Proof. unfold all_scopes. simpl. rewrite map_map. reflexivity. Qed.

Lemma in_all_levels_renamed (r : renaming) (tenv : Typing_env.t)
    (level : Typing_env_level.t) :
  level ∈ all_levels (rename_typing_env r tenv) ->
  ∃ level0 : Typing_env_level.t,
    rename_level r level0 = level ∧ level0 ∈ all_levels tenv.
Proof. rewrite all_levels_renamed. apply in_map_iff. Qed.

Lemma levels_disjoint_renamed (r : renaming)
    (earlier later : Typing_env_level.t) :
  levels_disjoint earlier later ->
  levels_disjoint (rename_level r earlier) (rename_level r later).
Proof.
  intros H. apply (proj1 (concat_defined _ _)).
  rewrite level_concat_equivariant.
  apply (proj2 (concat_defined _ _)) in H.
  destruct (Typing_env_level.concat earlier later);
    [ discriminate | elim H; reflexivity ].
Qed.

Lemma levels_pairwise_disjoint_renamed (r : renaming)
    (levels : list Typing_env_level.t) :
  levels_pairwise_disjoint levels ->
  levels_pairwise_disjoint (map (rename_level r) levels).
Proof.
  induction levels as [ | later levels IH]; simpl; [ trivial | ].
  intros [Hlater Hrest]. split; [ | exact (IH Hrest) ].
  intros earlier Hin. apply in_map_iff in Hin as (earlier0 & <- & Hin).
  exact (levels_disjoint_renamed r _ _ (Hlater _ Hin)).
Qed.

Lemma defined_variables_bound_renamed (r : renaming) (tenv : Typing_env.t) :
  defined_variables_bound tenv ->
  defined_variables_bound (rename_typing_env r tenv).
Proof.
  intros H level x k Hin Hlookup.
  apply in_all_levels_renamed in Hin as (level0 & <- & Hin).
  rewrite <- (rename_variable_inverse_r r x) in Hlookup |- *.
  pose proof
    (level_find_kind_equivariant r level0
       (rename_variable (inverse_renaming r) x))
    as Hkind.
  unfold Typing_env_level.find_kind in Hkind. rewrite Hkind in Hlookup.
  destruct (H _ _ _ Hin Hlookup)
    as (ty & binding_time_and_mode & Hty & Hk).
  exists (rename_type r ty), binding_time_and_mode. split.
  - exact (lookup_of_renamed r tenv _ _ _ Hty).
  - rewrite kind_rename_type. exact Hk.
Qed.

Lemma binding_times_renamed (r : renaming) (level : Typing_env_level.t)
    (binding_time : Binding_time.t) :
  Keyed.lookup Z.eqb binding_time
    (Typing_env_level.binding_times (rename_level r level))
  = option_map (map (rename_variable r))
      (Keyed.lookup Z.eqb binding_time
         (Typing_env_level.binding_times level)).
Proof. simpl. apply keyed_lookup_map_values. Qed.

Lemma indexed_at_renamed (r : renaming) (level : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) :
  indexed_at level binding_time x ->
  indexed_at (rename_level r level) binding_time (rename_variable r x).
Proof.
  intros (vars & Hlookup & Hin). exists (map (rename_variable r) vars). split.
  - rewrite binding_times_renamed, Hlookup. reflexivity.
  - exact (in_map _ _ _ Hin).
Qed.

Lemma indexed_at_renamed_back (r : renaming) (level : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) :
  indexed_at (rename_level r level) binding_time x ->
  indexed_at level binding_time (rename_variable (inverse_renaming r) x).
Proof.
  intros (vars & Hlookup & Hin). rewrite binding_times_renamed in Hlookup.
  destruct
    (Keyed.lookup Z.eqb binding_time (Typing_env_level.binding_times level))
    as [vars0 | ] eqn:Hvars0;
    [ | discriminate ].
  injection Hlookup as <-. apply in_map_iff in Hin as (x0 & <- & Hin).
  exists vars0. split; [ exact Hvars0 | ].
  rewrite rename_variable_inverse_l. exact Hin.
Qed.

Lemma level_indexed_renamed (r : renaming) (level : Typing_env_level.t) :
  level_indexed level -> level_indexed (rename_level r level).
Proof.
  intros H x. rewrite <- (rename_variable_inverse_r r x).
  generalize (rename_variable (inverse_renaming r) x). intros x0.
  pose proof (level_variable_is_defined_equivariant r level x0) as Hdefined.
  unfold Typing_env_level.variable_is_defined in Hdefined. rewrite Hdefined.
  split.
  - intros Hbound. apply (H x0) in Hbound as [binding_time Hindexed].
    exists binding_time. exact (indexed_at_renamed r _ _ _ Hindexed).
  - intros [binding_time Hindexed].
    apply indexed_at_renamed_back in Hindexed.
    rewrite rename_variable_inverse_l in Hindexed.
    apply (H x0). exists binding_time. exact Hindexed.
Qed.

Lemma levels_indexed_renamed (r : renaming) (tenv : Typing_env.t) :
  levels_indexed tenv -> levels_indexed (rename_typing_env r tenv).
Proof.
  intros H level Hin.
  apply in_all_levels_renamed in Hin as (level0 & <- & Hin).
  exact (level_indexed_renamed r _ (H _ Hin)).
Qed.

Lemma level_binding_times_recorded_renamed (r : renaming)
    (tenv : Typing_env.t) :
  level_binding_times_recorded tenv ->
  level_binding_times_recorded (rename_typing_env r tenv).
Proof.
  intros H level binding_time x Hin Hindexed.
  apply in_all_levels_renamed in Hin as (level0 & <- & Hin).
  apply indexed_at_renamed_back in Hindexed.
  destruct (H _ _ _ Hin Hindexed)
    as (ty & binding_time_and_mode & Hty & Htime).
  exists (rename_type r ty), binding_time_and_mode.
  split; [ | exact Htime ].
  rewrite <- (rename_variable_inverse_r r x).
  exact (lookup_of_renamed r tenv _ _ _ Hty).
Qed.

Lemma level_binding_times_in_range_renamed (r : renaming)
    (tenv : Typing_env.t) :
  level_binding_times_in_range tenv ->
  level_binding_times_in_range (rename_typing_env r tenv).
Proof.
  intros H level binding_time Hin Hbound.
  apply in_all_levels_renamed in Hin as (level0 & <- & Hin).
  apply (H level0 binding_time Hin). revert Hbound.
  unfold Keyed.is_bound. rewrite binding_times_renamed.
  destruct
    (Keyed.lookup Z.eqb binding_time
       (Typing_env_level.binding_times level0));
    simpl; intros Hbound; [ reflexivity | discriminate ].
Qed.

Lemma variables_defined_renamed (r : renaming) (tenv : Typing_env.t) :
  variables_defined tenv -> variables_defined (rename_typing_env r tenv).
Proof.
  intros H x ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & _ & Hlookup).
  destruct
    (H (rename_variable (inverse_renaming r) x) ty0 binding_time_and_mode
       Hlookup)
    as (level & Hin & Hbound).
  exists (rename_level r level). split.
  - rewrite all_levels_renamed. exact (in_map _ _ _ Hin).
  - rewrite <- (rename_variable_inverse_r r x).
    pose proof
      (level_variable_is_defined_equivariant r level
         (rename_variable (inverse_renaming r) x))
      as Hdefined.
    unfold Typing_env_level.variable_is_defined in Hdefined.
    rewrite Hdefined. exact Hbound.
Qed.

(* The aliases. *)

Lemma aliases_consistent_renamed (r : renaming) (t : Aliases.t) :
  aliases_consistent t -> aliases_consistent (rename_aliases r t).
Proof.
  intros [Hcanonical Hlisted]. split.
  - intros n canonical_element Hdemoted.
    apply demoted_to_renamed_inv in Hdemoted
      as (n' & canonical_element' & -> & -> & Hdemoted).
    unfold is_canonical. rewrite canonical_renamed.
    rewrite (Hcanonical n' canonical_element' Hdemoted). reflexivity.
  - intros n canonical_element. split.
    + intros Hdemoted.
      apply demoted_to_renamed_inv in Hdemoted
        as (n' & canonical_element' & -> & -> & Hdemoted).
      apply listed_alias_renamed, Hlisted, Hdemoted.
    + intros Hlisted'.
      rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n),
        <- (rename_simple_inverse_r r canonical_element)
        in Hlisted' |- *.
      apply listed_alias_renamed in Hlisted'.
      apply demoted_to_renamed, Hlisted, Hlisted'.
Qed.

Lemma name_mode_of_renamed (r : renaming) (tenv : Typing_env.t)
    (e : simple) :
  Aliases.compute_name_mode_unscoped
    (Typing_env.names_to_types (rename_typing_env r tenv))
    (rename_simple r e)
  = Aliases.compute_name_mode_unscoped (Typing_env.names_to_types tenv) e.
Proof.
  rewrite
    (compute_name_mode_unscoped_alike _
       (rename_binding_times r (Typing_env.names_to_types tenv))
       (times_of_renamed r tenv)).
  apply compute_name_mode_unscoped_renamed.
Qed.

Lemma name_modes_recorded_renamed (r : renaming) (tenv : Typing_env.t) :
  name_modes_recorded (Typing_env.names_to_types tenv)
    (Typing_env.aliases tenv) ->
  name_modes_recorded (Typing_env.names_to_types (rename_typing_env r tenv))
    (rename_aliases r (Typing_env.aliases tenv)).
Proof.
  intros Hrecorded canonical_element n name_mode Hlookup.
  rewrite <- (rename_simple_inverse_r r canonical_element) in Hlookup.
  rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n)
    in Hlookup |- *.
  rewrite get_aliases_of_canonical_element_renamed in Hlookup.
  unfold rename_aliases_of_canonical_element in Hlookup.
  rewrite
    (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r))
    in Hlookup.
  rewrite <- rename_simple_of_name, name_mode_of_renamed.
  apply (Hrecorded (rename_simple (inverse_renaming r) canonical_element)).
  destruct
    (Keyed.lookup name_eqb
       (rename_name (inverse_renaming r) n)
       (Aliases.get_aliases_of_canonical_element (Typing_env.aliases tenv)
          (rename_simple (inverse_renaming r) canonical_element)))
    as [name_mode' | ];
    [ | discriminate ].
  injection Hlookup as <-. reflexivity.
Qed.

Lemma aliases_known_renamed (r : renaming) (tenv : Typing_env.t) :
  aliases_known tenv -> aliases_known (rename_typing_env r tenv).
Proof.
  intros H n canonical_element Hdemoted.
  change (Typing_env.aliases (rename_typing_env r tenv))
    with (rename_aliases r (Typing_env.aliases tenv)) in Hdemoted.
  apply demoted_to_renamed_inv in Hdemoted
    as (n' & canonical_element' & -> & -> & Hdemoted).
  destruct (H _ _ Hdemoted) as [Hn Hcanonical]. split.
  - rewrite <- Hn. exact (mem_equivariant r None tenv n').
  - rewrite <- Hcanonical. apply mem_simple_equivariant.
Qed.

Lemma canonical_names_concrete_renamed (r : renaming) (tenv : Typing_env.t) :
  canonical_names_concrete tenv ->
  canonical_names_concrete (rename_typing_env r tenv).
Proof.
  intros H n ty binding_time_and_mode alias Hlookup Halias Hcanonical.
  apply lookup_in_renamed in Hlookup as (ty0 & -> & Hlookup).
  rewrite get_alias_opt_renamed in Halias.
  destruct (get_alias_opt ty0) as [alias0 | ] eqn:Halias0; [ | discriminate ].
  apply (H _ _ _ _ Hlookup Halias0).
  change (Typing_env.aliases (rename_typing_env r tenv))
    with (rename_aliases r (Typing_env.aliases tenv)) in Hcanonical.
  rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n)
    in Hcanonical.
  unfold is_canonical in Hcanonical |- *.
  rewrite simple_of_renamed_name, canonical_renamed in Hcanonical.
  destruct
    (Aliases.canonical (Typing_env.aliases tenv)
       (simple_of_name (rename_name (inverse_renaming r) n)));
    [ reflexivity | discriminate ].
Qed.

(* The equations of the levels. *)

Lemma all_equations_renamed (r : renaming) (tenv : Typing_env.t) :
  all_equations (rename_typing_env r tenv)
  = Keyed.map_bindings (rename_name r) (rename_type r) (all_equations tenv).
Proof.
  unfold all_equations. rewrite all_levels_renamed.
  induction (all_levels tenv) as [ | level levels IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite IH. unfold Keyed.map_bindings.
  rewrite map_app. reflexivity.
Qed.

Lemma levels_agree_renamed (r : renaming) (tenv : Typing_env.t) :
  levels_agree tenv -> levels_agree (rename_typing_env r tenv).
Proof.
  intros H n ty. rewrite all_equations_renamed.
  rewrite <- (FlambdaTypeRenameProofs.rename_name_inverse_r r n).
  rewrite lookup_named_renamed.
  destruct
    (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n)
       (all_equations tenv))
    as [ty0 | ] eqn:Hty0;
    [ | discriminate ].
  simpl. intros Heq. injection Heq as <-.
  destruct (H _ _ Hty0) as [binding_time_and_mode Hfound].
  exists binding_time_and_mode. exact (lookup_of_renamed r tenv _ _ _ Hfound).
Qed.

(* The conditions together. *)

Lemma conditions_renamed (r : renaming) (tenv : Typing_env.t) :
  typing_env_conditions tenv ->
  typing_env_conditions (rename_typing_env r tenv).
Proof.
  intros ((Hbounded & Hrange & Hdistinct & Hclosed & Hdefined & Hrecorded)
          & (Hdisjoint & Hbound & Hindexed & Htimes & Htimes_range & Hscopes
             & Hvariables)
          & (Hconsistent & Hmodes & Hknown & Hconcrete)).
  refine
    (conj (conj Hbounded (conj _ (conj _ (conj _ (conj _ _)))))
       (conj (conj _ (conj _ (conj _ (conj _ (conj _ (conj _ _))))))
          (conj _ (conj _ (conj _ _))))).
  - exact (binding_times_in_range_renamed r _ Hrange).
  - exact (binding_times_distinct_renamed r _ Hdistinct).
  - exact (types_closed_renamed r _ Hclosed).
  - exact (symbols_defined_renamed r _ Hdefined).
  - exact (symbols_recorded_renamed r _ Hrecorded).
  - rewrite all_levels_renamed.
    exact (levels_pairwise_disjoint_renamed r _ Hdisjoint).
  - exact (defined_variables_bound_renamed r _ Hbound).
  - exact (levels_indexed_renamed r _ Hindexed).
  - exact (level_binding_times_recorded_renamed r _ Htimes).
  - exact (level_binding_times_in_range_renamed r _ Htimes_range).
  - rewrite all_scopes_renamed. exact Hscopes.
  - exact (variables_defined_renamed r _ Hvariables).
  - exact (aliases_consistent_renamed r _ Hconsistent).
  - exact (name_modes_recorded_renamed r _ Hmodes).
  - exact (aliases_known_renamed r _ Hknown).
  - exact (canonical_names_concrete_renamed r _ Hconcrete).
Qed.

Lemma well_formed_renamed_true (r : renaming) (tenv : Typing_env.t) :
  typing_env_well_formed tenv = true ->
  typing_env_well_formed (rename_typing_env r tenv) = true.
Proof.
  intros H. apply (proj2 (typing_env_well_formed_iff _)), conditions_renamed.
  exact (proj1 (typing_env_well_formed_iff _) H).
Qed.

Theorem typing_env_well_formed_rename :
  typing_env_well_formed_rename_statement.
Proof.
  intros r tenv. destruct (typing_env_well_formed tenv) eqn:Htenv.
  - exact (well_formed_renamed_true r tenv Htenv).
  - destruct (typing_env_well_formed (rename_typing_env r tenv)) eqn:Hrenamed;
      [ | reflexivity ].
    apply (well_formed_renamed_true (inverse_renaming r)) in Hrenamed.
    rewrite rename_typing_env_inverse in Hrenamed. congruence.
Qed.

Theorem levels_agree_rename : levels_agree_rename_statement.
Proof.
  intros r tenv. split; [ | apply levels_agree_renamed ].
  intros H. apply (levels_agree_renamed (inverse_renaming r)) in H.
  rewrite rename_typing_env_inverse in H. exact H.
Qed.

Theorem typing_env_invariant_rename : typing_env_invariant_rename_statement.
Proof.
  intros r tenv. unfold typing_env_invariant.
  rewrite (typing_env_well_formed_rename r tenv).
  split; intros [Hwell_formed Hagree]; (split; [ exact Hwell_formed | ]);
    apply (levels_agree_rename r tenv); exact Hagree.
Qed.

Lemma types_well_formed_renamed (r : renaming) (tenv : Typing_env.t) :
  types_well_formed tenv -> types_well_formed (rename_typing_env r tenv).
Proof.
  intros H n ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & -> & Hlookup).
  rewrite (proj1 type_well_formed_rename). exact (H _ _ _ Hlookup).
Qed.

Theorem types_well_formed_rename : types_well_formed_rename_statement.
Proof.
  intros r tenv. split; [ | apply types_well_formed_renamed ].
  intros H. apply (types_well_formed_renamed (inverse_renaming r)) in H.
  rewrite rename_typing_env_inverse in H. exact H.
Qed.

Theorem types_well_formed_b_rename : types_well_formed_b_rename_statement.
Proof.
  intros r tenv. apply Bool.eq_true_iff_eq.
  exact
    (iff_trans (types_well_formed_b_iff (rename_typing_env r tenv))
       (iff_trans (types_well_formed_rename r tenv)
          (iff_sym (types_well_formed_b_iff tenv)))).
Qed.

(* ======================================================================== *)
(* The kinds of a renamed environment                                       *)
(* ======================================================================== *)

Lemma find_of_renamed (r : renaming) (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty : flambda_type) :
  Typing_env.find tenv n kind = Some ty ->
  Typing_env.find (rename_typing_env r tenv) (rename_name r n) kind
  = Some (rename_type r ty).
Proof. intros H. rewrite find_equivariant, H. reflexivity. Qed.

Lemma simple_of_kind_renamed (r : renaming) (tenv : Typing_env.t)
    (s : simple) (kind : flambda_kind) :
  simple_of_kind tenv s kind ->
  simple_of_kind (rename_typing_env r tenv) (rename_simple r s) kind.
Proof.
  destruct s as [x | sym | c]; simpl.
  - intros [ty Hfind]. exists (rename_type r ty).
    exact (find_of_renamed r tenv (Name_var x) (Some kind) ty Hfind).
  - intros [ty Hfind]. exists (rename_type r ty).
    exact (find_of_renamed r tenv (Name_symbol sym) (Some kind) ty Hfind).
  - trivial.
Qed.

Theorem simple_of_kind_rename : simple_of_kind_rename_statement.
Proof.
  intros r tenv s kind. split; [ | apply simple_of_kind_renamed ].
  intros H. apply (simple_of_kind_renamed (inverse_renaming r)) in H.
  rewrite rename_typing_env_inverse, rename_simple_inverse_l in H. exact H.
Qed.

Lemma types_kinded_renamed (r : renaming) (tenv : Typing_env.t) :
  types_kinded tenv -> types_kinded (rename_typing_env r tenv).
Proof.
  intros H n ty binding_time_and_mode Hlookup s kind Hin.
  apply lookup_in_renamed in Hlookup as (ty0 & -> & Hlookup).
  rewrite (proj1 aliases_with_kinds_rename_type) in Hin.
  apply in_map_iff in Hin as ([s0 kind0] & Heq & Hin). simpl in Heq.
  injection Heq as <- <-.
  exact (simple_of_kind_renamed r tenv _ _ (H _ _ _ Hlookup _ _ Hin)).
Qed.

Lemma class_canonical_element_renamed (r : renaming) (t : Aliases.t)
    (e : simple) :
  canonical_element_of (rename_aliases r t) (rename_simple r e)
  = rename_simple r (canonical_element_of t e).
Proof.
  unfold canonical_element_of. rewrite canonical_renamed.
  destruct (Aliases.canonical t e); reflexivity.
Qed.

Lemma aliased_renamed_back (r : renaming) (t : Aliases.t) (e e' : simple) :
  aliased (rename_aliases r t) (rename_simple r e) (rename_simple r e') ->
  aliased t e e'.
Proof.
  unfold aliased. rewrite !class_canonical_element_renamed. intros H.
  apply (f_equal (rename_simple (inverse_renaming r))) in H.
  rewrite !rename_simple_inverse_l in H. exact H.
Qed.

Lemma aliases_kinded_renamed (r : renaming) (tenv : Typing_env.t) :
  aliases_kinded tenv -> aliases_kinded (rename_typing_env r tenv).
Proof.
  intros H e e' kind kind' Haliased He He'.
  rewrite <- (rename_simple_inverse_r r e) in Haliased, He.
  rewrite <- (rename_simple_inverse_r r e') in Haliased, He'.
  change (Typing_env.aliases (rename_typing_env r tenv))
    with (rename_aliases r (Typing_env.aliases tenv)) in Haliased.
  apply aliased_renamed_back in Haliased.
  apply (proj1 (simple_of_kind_rename r tenv _ _)) in He.
  apply (proj1 (simple_of_kind_rename r tenv _ _)) in He'.
  exact (H _ _ _ _ Haliased He He').
Qed.

Lemma symbols_of_kind_value_renamed (r : renaming) (tenv : Typing_env.t) :
  symbols_of_kind_value tenv ->
  symbols_of_kind_value (rename_typing_env r tenv).
Proof.
  intros H s ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & -> & Hlookup).
  rewrite kind_rename_type. exact (H _ _ _ Hlookup).
Qed.

Lemma kinded_renamed (r : renaming) (tenv : Typing_env.t) :
  typing_env_kinded tenv -> typing_env_kinded (rename_typing_env r tenv).
Proof.
  intros (Htypes & Haliases & Hsymbols). split; [ | split ].
  - exact (types_kinded_renamed r _ Htypes).
  - exact (aliases_kinded_renamed r _ Haliases).
  - exact (symbols_of_kind_value_renamed r _ Hsymbols).
Qed.

Theorem typing_env_kinded_rename : typing_env_kinded_rename_statement.
Proof.
  intros r tenv. split; [ | apply kinded_renamed ].
  intros H. apply (kinded_renamed (inverse_renaming r)) in H.
  rewrite rename_typing_env_inverse in H. exact H.
Qed.

Lemma type_equations_kinded_renamed (r : renaming) (tenv : Typing_env.t)
    (ty : flambda_type) :
  type_equations_kinded tenv ty ->
  type_equations_kinded (rename_typing_env r tenv) (rename_type r ty).
Proof.
  intros H n kind Hin.
  rewrite (proj1 equation_names_rename_type) in Hin.
  apply in_map_iff in Hin as ([n0 kind0] & Heq & Hin). simpl in Heq.
  injection Heq as <- <-. rewrite simple_of_renamed_name.
  exact (simple_of_kind_renamed r tenv _ _ (H _ _ Hin)).
Qed.

Theorem type_equations_kinded_rename :
  type_equations_kinded_rename_statement.
Proof.
  intros r tenv ty. split; [ | apply type_equations_kinded_renamed ].
  intros H. apply (type_equations_kinded_renamed (inverse_renaming r)) in H.
  rewrite rename_typing_env_inverse, (proj1 rename_type_inverse) in H.
  exact H.
Qed.

Lemma types_equations_kinded_renamed (r : renaming) (tenv : Typing_env.t) :
  types_equations_kinded tenv ->
  types_equations_kinded (rename_typing_env r tenv).
Proof.
  intros H n ty binding_time_and_mode Hlookup.
  apply lookup_in_renamed in Hlookup as (ty0 & -> & Hlookup).
  exact (type_equations_kinded_renamed r tenv ty0 (H _ _ _ Hlookup)).
Qed.

Theorem types_equations_kinded_rename :
  types_equations_kinded_rename_statement.
Proof.
  intros r tenv. split; [ | apply types_equations_kinded_renamed ].
  intros H. apply (types_equations_kinded_renamed (inverse_renaming r)) in H.
  rewrite rename_typing_env_inverse in H. exact H.
Qed.
