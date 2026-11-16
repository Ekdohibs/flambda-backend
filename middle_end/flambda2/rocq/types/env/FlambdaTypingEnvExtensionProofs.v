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

(* Proofs of the statements of FlambdaTypingEnvExtensionSpec.v, and
   examples. Nothing here needs review. The examples are in the module
   [Examples] at the end: extensions made with the functions, and the
   environments that do and do not satisfy them. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaBindingTime
  FlambdaEnvAliasesRenamingProofs FlambdaEnvAliasesSetProofs FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvExtension FlambdaTypingEnvExtensionSpec.
Import ListNotations.

(* ======================================================================== *)
(* Maps                                                                     *)
(* ======================================================================== *)

Lemma flat_map_ext_in {B C : Type} (f g : B -> list C) (l : list B) :
  (∀ b : B, b ∈ l -> f b = g b) -> flat_map f l = flat_map g l.
Proof.
  induction l as [ | b l IH]; intros Hsame; simpl; [ reflexivity | ].
  rewrite (Hsame b (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b' Hin. apply Hsame. right. exact Hin.
Qed.

Section Keys.

Context {K : Type} (eqb : K -> K -> bool) (Heqb : decides_equality eqb).

Lemma eqb_refl (k : K) : eqb k k = true.
Proof. apply Heqb. reflexivity. Qed.

Lemma eqb_sym (k k' : K) : eqb k k' = eqb k' k.
Proof.
  destruct (eqb k k') eqn:Hfirst, (eqb k' k) eqn:Hsecond; try reflexivity.
  - apply Heqb in Hfirst. subst k'. rewrite eqb_refl in Hsecond.
    discriminate.
  - apply Heqb in Hsecond. subst k'. rewrite eqb_refl in Hfirst.
    discriminate.
Qed.

Lemma without_not_among (k : K) (keys : list K) :
  Keyed.among eqb k keys = false -> Keyed.without eqb [k] keys = keys.
Proof.
  unfold Keyed.without, Keyed.among.
  induction keys as [ | key keys IH]; simpl; [ reflexivity | ].
  intros Hnot. apply orb_false_iff in Hnot as [Hkey Hkeys].
  rewrite eqb_sym, Hkey. simpl. f_equal. apply IH, Hkeys.
Qed.

Lemma dedup_of_distinct (keys : list K) :
  Keyed.distinct eqb keys = true -> Keyed.dedup eqb keys = keys.
Proof.
  induction keys as [ | k keys IH]; [ reflexivity | ].
  cbn [Keyed.distinct Keyed.dedup]. intros Hdistinct.
  apply andb_true_iff in Hdistinct as [Hk Hkeys]. apply negb_true_iff in Hk.
  rewrite (IH Hkeys), (without_not_among k keys Hk). reflexivity.
Qed.

Lemma tabulate_of_distinct {B : Type} (bindings : Keyed.finite_map K B) :
  Keyed.distinct eqb (map fst bindings) = true ->
  Keyed.tabulate (fun key => Keyed.lookup eqb key bindings) (map fst bindings)
  = bindings.
Proof.
  unfold Keyed.tabulate.
  induction bindings as [ | [k b] bindings IH]; simpl; intros Hdistinct;
    [ reflexivity | ].
  apply andb_true_iff in Hdistinct as [Hk Hdistinct].
  apply negb_true_iff in Hk. rewrite eqb_refl. simpl. f_equal.
  transitivity
    (flat_map
       (fun key =>
          match Keyed.lookup eqb key bindings with
          | Some a => [(key, a)]
          | None => []
          end)
       (map fst bindings));
    [ | exact (IH Hdistinct) ].
  apply flat_map_ext_in. intros key Hin.
  destruct (eqb key k) eqn:Hequal; [ | reflexivity ].
  apply Heqb in Hequal. subst key.
  apply (keyed_among_in K eqb Heqb) in Hin. congruence.
Qed.

(* A map in which no key has two bindings is its own compaction. *)
Lemma compact_of_distinct {B : Type} (bindings : Keyed.finite_map K B) :
  Keyed.distinct eqb (map fst bindings) = true ->
  Keyed.compact eqb bindings = bindings.
Proof.
  unfold Keyed.compact, Keyed.domain. intros Hdistinct.
  rewrite (dedup_of_distinct _ Hdistinct).
  apply tabulate_of_distinct, Hdistinct.
Qed.

Lemma keys_unbind {B : Type} (k : K) (bindings : Keyed.finite_map K B) :
  map fst (Keyed.unbind eqb k bindings)
  = filter (fun key => negb (eqb k key)) (map fst bindings).
Proof.
  unfold Keyed.unbind.
  induction bindings as [ | [k' b] bindings IH]; simpl; [ reflexivity | ].
  destruct (eqb k k'); simpl; rewrite IH; reflexivity.
Qed.

(* Two maps have no key in common. *)
Lemma disjoint_domains {B : Type} (bindings bindings' : Keyed.finite_map K B) :
  Keyed.is_empty
    (Keyed.inter eqb (Keyed.domain eqb bindings) (Keyed.domain eqb bindings'))
  = true
  <-> (∀ k : K,
        Keyed.is_bound eqb k bindings = true ->
        Keyed.is_bound eqb k bindings' = false).
Proof.
  rewrite (keyed_is_empty_among K eqb Heqb). split.
  - intros Hnone k Hbound. specialize (Hnone k).
    rewrite (keyed_among_inter K eqb Heqb), !(keyed_among_domain K eqb Heqb),
      Hbound
      in Hnone.
    exact Hnone.
  - intros Hdisjoint k.
    rewrite (keyed_among_inter K eqb Heqb), !(keyed_among_domain K eqb Heqb).
    destruct (Keyed.is_bound eqb k bindings) eqn:Hbound; [ | reflexivity ].
    rewrite (Hdisjoint k Hbound). reflexivity.
Qed.

End Keys.

(* ======================================================================== *)
(* The equations                                                            *)
(* ======================================================================== *)

Lemma satisfied_lookup (statics : static_env) (vars : env)
    (t : Typing_env_extension.t) :
  env_extension_satisfied statics vars t
  <-> (∀ (n : name) (ty : flambda_type),
        Keyed.lookup name_eqb n (Typing_env_extension.to_map t) = Some ty ->
        name_has_type statics vars n ty).
Proof. destruct t as [equations]. apply env_extension_lookup. Qed.

Theorem to_map_from_map : to_map_from_map_statement.
Proof. split; [ reflexivity | intros [equations]; reflexivity ]. Qed.

Theorem bindings_lookup : bindings_lookup_statement.
Proof. intros t n ty. apply (keyed_in_compact name_eqb name_eqb_eq). Qed.

Theorem bindings_distinct : bindings_distinct_statement.
Proof.
  intros t. apply (keyed_compact_distinct name name_eqb name_eqb_eq).
Qed.

Theorem bindings_of_distinct : bindings_of_distinct_statement.
Proof. intros t. apply (compact_of_distinct name_eqb name_eqb_eq). Qed.

Theorem is_empty_lookup : is_empty_lookup_statement.
Proof.
  intros t. apply (keyed_is_empty_lookup name name_eqb name_eqb_eq).
Qed.

Theorem has_equation_lookup : has_equation_lookup_statement.
Proof. intros t n. apply keyed_is_bound_lookup. Qed.

(* ======================================================================== *)
(* Making an extension                                                      *)
(* ======================================================================== *)

Lemma name_eqb_refl (n : name) : name_eqb n n = true.
Proof. apply name_eqb_eq. reflexivity. Qed.

Theorem empty_satisfied : empty_satisfied_statement.
Proof.
  intros statics vars. apply satisfied_lookup. intros n ty Hlookup.
  discriminate Hlookup.
Qed.

Theorem one_equation_satisfied : one_equation_satisfied_statement.
Proof.
  intros statics vars n ty. rewrite satisfied_lookup. split.
  - intros Hall. apply Hall. simpl. rewrite name_eqb_refl. reflexivity.
  - intros Hhas n' ty' Hlookup. simpl in Hlookup.
    destruct (name_eqb n' n) eqn:Hequal; [ | discriminate ].
    apply name_eqb_eq in Hequal. subst n'. injection Hlookup as <-.
    exact Hhas.
Qed.

Lemma lookup_added (equations : Keyed.finite_map name flambda_type)
    (n : name) (ty : flambda_type) (n' : name) :
  Keyed.lookup name_eqb n'
    ((n, ty) :: Keyed.unbind name_eqb n equations)
  = if name_eqb n' n then Some ty else Keyed.lookup name_eqb n' equations.
Proof.
  rewrite keyed_lookup_cons.
  destruct (name_eqb n' n) eqn:Hequal; [ reflexivity | ].
  rewrite (keyed_lookup_unbind name name_eqb name_eqb_eq), Hequal.
  reflexivity.
Qed.

Theorem add_or_replace_equation_lookup :
  add_or_replace_equation_lookup_statement.
Proof. intros t n ty n'. apply lookup_added. Qed.

Theorem add_or_replace_equation_distinct :
  add_or_replace_equation_distinct_statement.
Proof.
  intros t n ty Hdistinct.
  unfold Typing_env_extension.add_or_replace_equation. simpl.
  apply andb_true_iff. split.
  - apply negb_true_iff.
    destruct
      (Keyed.among name_eqb n
         (map fst
            (Keyed.unbind name_eqb n (Typing_env_extension.to_map t))))
      eqn:Hamong;
      [ | reflexivity ].
    apply (keyed_among_in name name_eqb name_eqb_eq) in Hamong.
    exfalso. revert Hamong.
    apply (keyed_lookup_none_not_in name_eqb name_eqb_eq).
    rewrite (keyed_lookup_unbind name name_eqb name_eqb_eq), name_eqb_refl.
    reflexivity.
  - rewrite keys_unbind.
    apply (keyed_distinct_filter name_eqb name_eqb_eq), Hdistinct.
Qed.

Theorem add_or_replace_equation_satisfied :
  add_or_replace_equation_satisfied_statement.
Proof.
  intros statics vars t n ty. rewrite satisfied_lookup. split.
  - intros Hall. split.
    + apply Hall. rewrite add_or_replace_equation_lookup, name_eqb_refl.
      reflexivity.
    + intros n' ty' Hdifferent Hlookup. apply Hall.
      rewrite add_or_replace_equation_lookup.
      destruct (name_eqb n' n) eqn:Hequal; [ | exact Hlookup ].
      apply name_eqb_eq in Hequal. contradiction.
  - intros [Hhas Hothers] n' ty' Hlookup.
    rewrite add_or_replace_equation_lookup in Hlookup.
    destruct (name_eqb n' n) eqn:Hequal.
    + apply name_eqb_eq in Hequal. subst n'. injection Hlookup as <-.
      exact Hhas.
    + apply Hothers; [ | exact Hlookup ].
      intros ->. rewrite name_eqb_refl in Hequal. discriminate.
Qed.

Theorem replace_equation : replace_equation_statement.
Proof. intros t n ty. reflexivity. Qed.

Theorem disjoint_union_defined : disjoint_union_defined_statement.
Proof.
  intros t1 t2.
  unfold Typing_env_extension.disjoint_union, Typing_env_extension.has_equation.
  rewrite <- (disjoint_domains name_eqb name_eqb_eq).
  destruct
    (Keyed.is_empty
       (Keyed.inter name_eqb
          (Keyed.domain name_eqb (Typing_env_extension.to_map t1))
          (Keyed.domain name_eqb (Typing_env_extension.to_map t2))));
    split; try discriminate; try reflexivity; intros H; congruence.
Qed.

Theorem disjoint_union_satisfied : disjoint_union_satisfied_statement.
Proof.
  intros statics vars t1 t2 t.
  unfold Typing_env_extension.disjoint_union.
  destruct
    (Keyed.is_empty
       (Keyed.inter name_eqb
          (Keyed.domain name_eqb (Typing_env_extension.to_map t1))
          (Keyed.domain name_eqb (Typing_env_extension.to_map t2))))
    eqn:Hdisjoint;
    [ | discriminate ].
  pose proof (proj1 (disjoint_domains name_eqb name_eqb_eq _ _) Hdisjoint)
    as Hapart.
  clear Hdisjoint. rename Hapart into Hdisjoint.
  intros [= <-]. rewrite !satisfied_lookup. simpl. split.
  - intros Hall. split.
    + intros n ty Hlookup. apply Hall. rewrite keyed_lookup_app, Hlookup.
      reflexivity.
    + intros n ty Hlookup. apply Hall. rewrite keyed_lookup_app.
      destruct (Keyed.lookup name_eqb n (Typing_env_extension.to_map t1))
        as [ty1 | ] eqn:Hfirst;
        [ | exact Hlookup ].
      assert
        (Hbound
          : Keyed.is_bound name_eqb n (Typing_env_extension.to_map t2) = true).
      { apply keyed_is_bound_lookup. exists ty. exact Hlookup. }
      rewrite Hdisjoint in Hbound; [ discriminate | ].
      apply keyed_is_bound_lookup. exists ty1. exact Hfirst.
  - intros [Hfirst Hsecond] n ty Hlookup.
    rewrite keyed_lookup_app in Hlookup.
    destruct (Keyed.lookup name_eqb n (Typing_env_extension.to_map t1))
      as [ty1 | ] eqn:Hin_first.
    + injection Hlookup as <-. apply Hfirst, Hin_first.
    + apply Hsecond, Hlookup.
Qed.

(* ======================================================================== *)
(* An extension taken from an environment                                   *)
(* ======================================================================== *)

Theorem environment_extension_satisfied :
  environment_extension_satisfied_statement.
Proof.
  intros statics valuation tenv t (_ & Hnames & _) Hfound Hvalues.
  apply satisfied_lookup. intros n ty Hlookup.
  destruct (Hfound n ty Hlookup) as [binding_time_and_mode Hfind].
  destruct (Hvalues n ty Hlookup) as [v Hvalue].
  exists v. split; [ exact Hvalue | ].
  destruct (Hnames n ty binding_time_and_mode v Hfind Hvalue)
    as [[[] _] | Hsatisfied].
  exact Hsatisfied.
Qed.

(* The equation that the extension made of a level has for a name. *)
Lemma lookup_level_extension (level : Typing_env_level.t) (n : name) :
  Keyed.lookup name_eqb n
    (Typing_env_extension.to_map
       (Typing_env_level.as_extension_without_bindings level))
  = Keyed.lookup name_eqb n (Typing_env_level.equations level).
Proof.
  unfold Typing_env_level.as_extension_without_bindings. simpl.
  try rewrite (keyed_lookup_compact name name_eqb name_eqb_eq).
  reflexivity.
Qed.

Theorem level_extension_satisfied : level_extension_satisfied_statement.
Proof.
  intros statics valuation tenv level Hsatisfied Hfound Hvalues.
  apply (environment_extension_satisfied statics valuation tenv);
    [ exact Hsatisfied | | ];
    intros n ty Hlookup; rewrite lookup_level_extension in Hlookup.
  - apply Hfound, Hlookup.
  - apply (Hvalues n ty), Hlookup.
Qed.

(* ======================================================================== *)
(* Extensions with extra variables                                          *)
(* ======================================================================== *)

Theorem with_extra_variables_empty : with_extra_variables_empty_statement.
Proof.
  intros statics vars. exists vars. split; [ reflexivity | ].
  apply (empty_satisfied statics vars).
Qed.

Theorem with_no_extra_variables : with_no_extra_variables_statement.
Proof.
  intros statics vars equations. split.
  - intros [vars' [Hagree Hsatisfied]]. simpl in Hagree, Hsatisfied.
    apply
      (env_extension_satisfied_coincide statics vars' vars
         (Env_extension equations));
      [ | exact Hsatisfied ].
    intros x _. apply Hagree. intros [].
  - intros Hsatisfied. exists vars. split; [ reflexivity | exact Hsatisfied ].
Qed.

Theorem add_definition_satisfied : add_definition_satisfied_statement.
Proof.
  intros statics vars t var kind [vars' [Hagree Hsatisfied]].
  exists vars'. split; [ | exact Hsatisfied ].
  intros x Hnot. apply Hagree. intros Hin. apply Hnot. right. exact Hin.
Qed.

Theorem with_extra_variables_add_or_replace_equation :
  with_extra_variables_add_or_replace_equation_statement.
Proof.
  intros t n ty n'. split; [ reflexivity | apply lookup_added ].
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem apply_renaming_to_map : apply_renaming_to_map_statement.
Proof. intros r [equations]. reflexivity. Qed.

Lemma compact_renamed (r : renaming)
    (m : Keyed.finite_map name flambda_type) :
  Keyed.compact name_eqb
    (Keyed.map_bindings (rename_name r) (rename_type r) m)
  = Keyed.map_bindings (rename_name r) (rename_type r)
      (Keyed.compact name_eqb m).
Proof. apply compact_keys_renamed. exact (name_eqb_renamed r). Qed.

Theorem apply_renaming_bindings : apply_renaming_bindings_statement.
Proof.
  intros r t. unfold Typing_env_extension.bindings.
  rewrite (apply_renaming_to_map r t). apply compact_renamed.
Qed.

Theorem apply_renaming_is_empty : apply_renaming_is_empty_statement.
Proof.
  intros r t. unfold Typing_env_extension.is_empty.
  rewrite (apply_renaming_to_map r t). apply is_empty_map.
Qed.

Theorem apply_renaming_has_equation : apply_renaming_has_equation_statement.
Proof.
  intros r t n. unfold Typing_env_extension.has_equation.
  rewrite (apply_renaming_to_map r t).
  apply (is_bound_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
Qed.

Theorem apply_renaming_one_equation : apply_renaming_one_equation_statement.
Proof. intros r n ty. reflexivity. Qed.

Theorem apply_renaming_add_or_replace_equation :
  apply_renaming_add_or_replace_equation_statement.
Proof.
  intros r [equations] n ty.
  change
    (Env_extension
       (Keyed.map_bindings (rename_name r) (rename_type r)
          ((n, ty) :: Keyed.unbind name_eqb n equations))
     = Env_extension
         ((rename_name r n, rename_type r ty)
          :: Keyed.unbind name_eqb (rename_name r n)
               (Keyed.map_bindings (rename_name r) (rename_type r)
                  equations))).
  rewrite (unbind_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  reflexivity.
Qed.

Theorem apply_renaming_disjoint_union :
  apply_renaming_disjoint_union_statement.
Proof.
  intros r t1 t2. unfold Typing_env_extension.disjoint_union.
  rewrite (apply_renaming_to_map r t1), (apply_renaming_to_map r t2).
  rewrite !(domain_renamed name_eqb (rename_name r) (name_eqb_renamed r)),
    inter_renamed, is_empty_map.
  destruct (Keyed.is_empty _); [ | reflexivity ].
  destruct t1 as [equations1], t2 as [equations2]. simpl.
  unfold Keyed.map_bindings. rewrite map_app. reflexivity.
Qed.

Theorem apply_renaming_fold : apply_renaming_fold_statement.
Proof.
  intros B B' r state equation equation' t acc Hequation.
  unfold Typing_env_extension.fold.
  rewrite (apply_renaming_bindings r t).
  apply
    (fold_left_renamed state
       (fun binding : name * flambda_type =>
          (rename_name r (fst binding), rename_type r (snd binding)))).
  intros a [n ty]. apply Hequation.
Qed.

Lemma get_alias_opt_renamed (r : renaming) (ty : flambda_type) :
  get_alias_opt (rename_type r ty)
  = option_map (rename_simple r) (get_alias_opt ty).
Proof.
  destruct ty as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; reflexivity.
Qed.

Lemma check_equation_renamed (r : renaming) (n : name) (ty : flambda_type) :
  Typing_env_extension.check_equation (rename_name r n) (rename_type r ty)
  = Typing_env_extension.check_equation n ty.
Proof.
  unfold Typing_env_extension.check_equation,
    Typing_env_extension.is_alias_of_name.
  rewrite get_alias_opt_renamed.
  destruct (get_alias_opt ty) as [[x | sym | c] | ]; simpl;
    try reflexivity.
  - change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
    rewrite name_eqb_renamed. reflexivity.
  - change (Name_symbol sym) with (rename_name r (Name_symbol sym)) at 1.
    rewrite name_eqb_renamed. reflexivity.
Qed.

Theorem apply_renaming_checks : apply_renaming_checks_statement.
Proof.
  split; [ exact check_equation_renamed | ].
  intros r t. unfold Typing_env_extension.invariant.
  rewrite (apply_renaming_bindings r t).
  apply forallb_renamed. intros [n ty]. apply check_equation_renamed.
Qed.

Theorem with_extra_variables_rename_identity :
  with_extra_variables_rename_identity_statement.
Proof.
  intros [vars equations].
  unfold Typing_env_extension.With_extra_variables.apply_renaming.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  f_equal.
  - apply map_fixes. intros x _. reflexivity.
  - apply map_bindings_identity;
      [ apply rename_name_identity | apply (proj1 rename_type_identity) ].
Qed.

Theorem with_extra_variables_rename_compose :
  with_extra_variables_rename_compose_statement.
Proof.
  intros second first [vars equations].
  unfold Typing_env_extension.With_extra_variables.apply_renaming.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  f_equal.
  - rewrite map_map. apply map_ext. intros x.
    apply rename_variable_compose.
  - rewrite map_bindings_compose.
    apply map_bindings_extensional;
      [ intros n; apply rename_name_compose
      | intros ty; apply (proj1 rename_type_compose) ].
Qed.

Theorem with_extra_variables_free_names_renamed :
  with_extra_variables_free_names_renamed_statement.
Proof.
  intros r [vars equations].
  unfold Typing_env_extension.With_extra_variables.free_names,
    Typing_env_extension.With_extra_variables.apply_renaming.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  rewrite map_app. f_equal.
  - rewrite !map_map. reflexivity.
  - exact (proj2 free_names_rename_type r (Env_extension equations)).
Qed.

Theorem with_extra_variables_rename_fixes :
  with_extra_variables_rename_fixes_statement.
Proof.
  intros r [vars equations] Hfixes.
  unfold Typing_env_extension.With_extra_variables.free_names in Hfixes.
  unfold Typing_env_extension.With_extra_variables.apply_renaming.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations] in *.
  f_equal.
  - apply map_fixes. intros x Hx. apply Hfixes, in_or_app. left.
    apply in_map, Hx.
  - assert (Hextension :
      rename_env_extension r (Env_extension equations)
      = Env_extension equations).
    { apply (proj2 rename_type_fixes). intros x Hx.
      apply Hfixes, in_or_app. right.
      rewrite (proj2 free_names_variables (Env_extension equations)) in Hx.
      apply in_flat_map in Hx. destruct Hx as (n & Hn & Hx).
      destruct n as [y | sym]; simpl in Hx; [ | contradiction ].
      destruct Hx as [<- | []]. exact Hn. }
    injection Hextension. auto.
Qed.

Theorem with_extra_variables_rename_add_definition :
  with_extra_variables_rename_add_definition_statement.
Proof. intros r t var kind. reflexivity. Qed.

Theorem with_extra_variables_rename_add_equation :
  with_extra_variables_rename_add_equation_statement.
Proof.
  intros r [vars equations] n ty.
  unfold Typing_env_extension.With_extra_variables.apply_renaming,
    Typing_env_extension.With_extra_variables.add_or_replace_equation.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  f_equal.
  rewrite (unbind_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  reflexivity.
Qed.

Theorem with_extra_variables_rename_map_types :
  with_extra_variables_rename_map_types_statement.
Proof.
  intros r f f' [vars equations] Hf.
  unfold Typing_env_extension.With_extra_variables.apply_renaming,
    Typing_env_extension.With_extra_variables.map_types.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  f_equal. unfold Keyed.map_bindings, Keyed.map_values.
  rewrite !map_map. apply map_ext. intros [n ty]. simpl. rewrite Hf.
  reflexivity.
Qed.

Theorem with_extra_variables_rename_fold :
  with_extra_variables_rename_fold_statement.
Proof.
  intros B B' r state variable variable' equation equation'
    [vars equations] acc Hvariable Hequation.
  unfold Typing_env_extension.With_extra_variables.fold,
    Typing_env_extension.With_extra_variables.apply_renaming.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  cbv zeta.
  rewrite compact_renamed,
    (fold_left_renamed state (rename_variable r)
       (fun acc var => variable var acc) (fun acc var => variable' var acc)).
  - apply
      (fold_left_renamed state
         (fun binding : name * flambda_type =>
            (rename_name r (fst binding), rename_type r (snd binding)))).
    intros a [n ty]. apply Hequation.
  - intros a x. apply Hvariable.
Qed.

Theorem with_extra_variables_satisfied_rename :
  with_extra_variables_satisfied_rename_statement.
Proof.
  intros r statics vars [extra equations].
  unfold with_extra_variables_satisfied,
    Typing_env_extension.With_extra_variables.apply_renaming.
  cbn [Typing_env_extension.With_extra_variables.existential_vars
       Typing_env_extension.With_extra_variables.equations].
  split.
  - intros (vars' & Hagree & Hsatisfied).
    exists (rename_env (inverse_renaming r) vars'). split.
    + intros x Hx.
      rewrite <- (rename_variable_inverse_l r x) at 1.
      rewrite lookup_rename_env, Hagree; [ apply lookup_rename_env | ].
      intros Hin. apply in_map_iff in Hin. destruct Hin as (y & Heq & Hy).
      apply rename_variable_injective in Heq. subst y. exact (Hx Hy).
    + pose proof
        (proj2
           (env_extension_satisfied_rename (inverse_renaming r) statics vars'
              (rename_env_extension r (Env_extension equations)))
           Hsatisfied)
        as H.
      rewrite (proj2 rename_type_inverse r (Env_extension equations)) in H.
      exact H.
  - intros (vars' & Hagree & Hsatisfied).
    exists (rename_env r vars'). split.
    + intros x Hx.
      rewrite <- (rename_variable_inverse_r r x), !lookup_rename_env.
      apply Hagree. intros Hin. apply Hx, in_map_iff.
      exists (rename_variable (inverse_renaming r) x).
      split; [ apply rename_variable_inverse_r | exact Hin ].
    + exact
        (proj2
           (env_extension_satisfied_rename r statics vars'
              (Env_extension equations))
           Hsatisfied).
Qed.

Theorem with_extra_variables_bound : with_extra_variables_bound_statement.
Proof.
  intros statics vars vars' t Hsame (vars'' & Hagree & Hsatisfied).
  exists vars''. split; [ | exact Hsatisfied ].
  intros x Hx. rewrite (Hagree x Hx). symmetry. exact (Hsame x Hx).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Local Open Scope Z_scope.

Definition w : variable := 1%positive.
Definition x : variable := 2%positive.
Definition y : variable := 3%positive.

Definition statics : static_env := Static_env [] [].

Definition tagged (i : Z) : value := Value_const (Tagged_immediate i).

(* [x] is 5 or 6. *)
Definition x_is_five_or_six : Typing_env_extension.t :=
  Typing_env_extension.one_equation (Name_var x)
    (these_tagged_immediates [5; 6]).

Example satisfied_at_five :
  env_extension_satisfied statics [(x, tagged 5)] x_is_five_or_six.
Proof.
  apply one_equation_satisfied. exists (tagged 5).
  split; [ reflexivity | ].
  apply these_tagged_immediates_satisfied. exists 5.
  split; [ reflexivity | left; reflexivity ].
Qed.

Example not_satisfied_at_seven :
  ~ env_extension_satisfied statics [(x, tagged 7)] x_is_five_or_six.
Proof.
  intros Hsatisfied.
  apply one_equation_satisfied in Hsatisfied as (v & Hvalue & Htype).
  injection Hvalue as <-.
  apply these_tagged_immediates_satisfied in Htype as (i & Hi & Hin).
  injection Hi as <-. destruct Hin as [H | [H | []]]; discriminate H.
Qed.

(* A name with an equation has to have a value. *)
Example not_satisfied_without_a_value :
  ~ env_extension_satisfied statics [] x_is_five_or_six.
Proof.
  intros Hsatisfied.
  apply one_equation_satisfied in Hsatisfied as (v & Hvalue & _).
  discriminate.
Qed.

(* A second equation on [x] replaces the first. *)
Definition x_is_five : Typing_env_extension.t :=
  Typing_env_extension.add_or_replace_equation x_is_five_or_six (Name_var x)
    (these_tagged_immediates [5]).

Example one_equation_for_a_name :
  Typing_env_extension.bindings x_is_five
  = [(Name_var x, these_tagged_immediates [5])].
Proof. reflexivity. Qed.

(* [y] is an alias of [x]. *)
Definition x_and_y : Typing_env_extension.t :=
  Typing_env_extension.add_or_replace_equation x_is_five (Name_var y)
    (alias_type_of Value (Var x)).

(* [fold] comes to the equation added last first. *)
Example names_as_fold_comes_to_them :
  Typing_env_extension.fold (fun n _ names => names ++ [n]) x_and_y []
  = [Name_var y; Name_var x].
Proof. reflexivity. Qed.

Example both_equations_satisfied :
  env_extension_satisfied statics [(x, tagged 5); (y, tagged 5)] x_and_y.
Proof.
  apply add_or_replace_equation_satisfied. split.
  - exists (tagged 5). split; [ reflexivity | ].
    apply alias_satisfied. split; reflexivity.
  - intros n ty Hdifferent Hlookup.
    change (Typing_env_extension.to_map x_is_five)
      with [(Name_var x, these_tagged_immediates [5])]
      in Hlookup.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    destruct Hlookup as [[= <- <-] | []]. exists (tagged 5).
    split; [ reflexivity | ].
    apply (proj2 (these_tagged_immediates_satisfied statics _ (tagged 5) [5])).
    exists 5. split; [ reflexivity | left; reflexivity ].
Qed.

Example alias_not_satisfied :
  ~ env_extension_satisfied statics [(x, tagged 5); (y, tagged 6)] x_and_y.
Proof.
  intros Hsatisfied.
  apply add_or_replace_equation_satisfied in Hsatisfied
    as [(v & Hvalue & Htype) _].
  injection Hvalue as <-. apply alias_satisfied in Htype as [Heq _].
  discriminate.
Qed.

(* Two extensions with no name in common, and two with one. *)
Example union_of_two :
  Typing_env_extension.disjoint_union x_is_five
    (Typing_env_extension.one_equation (Name_var y)
       (alias_type_of Value (Var x)))
  = Some
      (Typing_env_extension.from_map
         [(Name_var x, these_tagged_immediates [5]);
          (Name_var y, alias_type_of Value (Var x))]).
Proof. reflexivity. Qed.

Example no_union_of_two_on_one_name :
  Typing_env_extension.disjoint_union x_is_five x_is_five_or_six = None.
Proof. reflexivity. Qed.

(* An extension with an extra variable: [y] is an alias of some [w]. An
   environment in which [y] has a value satisfies it, with that value
   for [w]; as a plain extension it is not satisfied there, since [w]
   has no value. *)
Definition y_is_some_w : Typing_env_extension.With_extra_variables.t :=
  Typing_env_extension.With_extra_variables.add_or_replace_equation
    (Typing_env_extension.With_extra_variables.add_definition
       Typing_env_extension.With_extra_variables.empty w Value)
    (Name_var y)
    (alias_type_of Value (Var w)).

Example extra_variable_satisfied :
  with_extra_variables_satisfied statics [(y, tagged 5)] y_is_some_w.
Proof.
  exists [(w, tagged 5); (y, tagged 5)]. split.
  - intros z Hnot.
    change (lookup z [(w, tagged 5); (y, tagged 5)])
      with (if Pos.eqb z w then Some (tagged 5) else lookup z [(y, tagged 5)]).
    destruct (Pos.eqb_spec z w) as [-> | Hne]; [ | reflexivity ].
    exfalso. apply Hnot. left. reflexivity.
  - apply
      (one_equation_satisfied statics _ (Name_var y)
         (alias_type_of Value (Var w))).
    exists (tagged 5). split; [ reflexivity | ].
    apply alias_satisfied. split; reflexivity.
Qed.

Example plain_extension_not_satisfied :
  ~ env_extension_satisfied statics [(y, tagged 5)]
      (Env_extension
         (Typing_env_extension.With_extra_variables.equations y_is_some_w)).
Proof.
  intros Hsatisfied.
  apply
    (one_equation_satisfied statics _ (Name_var y)
       (alias_type_of Value (Var w)))
    in Hsatisfied as (v & Hvalue & Htype).
  apply alias_satisfied in Htype as [Heq _]. discriminate.
Qed.

End Examples.
