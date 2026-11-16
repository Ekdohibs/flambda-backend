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


(* Proofs of the statements of FlambdaTypingEnvKindedSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeAliasKindsSpec
  FlambdaTypeAliasKindsProofs FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvKindedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The kind of a [Simple]                                                   *)
(* ======================================================================== *)

(* [simple_of_kind] of a name, as one proposition. *)
Lemma simple_of_kind_of_name (tenv : Typing_env.t) (n : name)
    (kind : flambda_kind) :
  simple_of_kind tenv (simple_of_name n) kind
  <-> ∃ ty : flambda_type, Typing_env.find tenv n (Some kind) = Some ty.
Proof. destruct n; reflexivity. Qed.

Theorem simple_of_kind_name : simple_of_kind_name_statement.
Proof.
  intros tenv n kind. rewrite simple_of_kind_of_name.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode',
    Typing_env.mem.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty' btm] | ] eqn:Hlookup.
  - cbn [Typing_env.check_optional_kind_matches].
    destruct (kind_eqb kind (FlambdaTypeGrammar.kind ty')) eqn:Hkind.
    + apply kind_eqb_eq in Hkind. split.
      * intros _. left. exists ty', btm.
        split; [ reflexivity | symmetry; exact Hkind ].
      * intros _. destruct (Typing_env.is_bottom tenv); eexists; reflexivity.
    + split.
      * intros [ty Hfind]. discriminate.
      * intros [[ty [btm' [[= <- <-] Hk]]] | [Hnone _]]; [ | discriminate ].
        subst kind. rewrite (proj2 (kind_eqb_eq _ _) eq_refl) in Hkind.
        discriminate.
  - destruct n as [x | s].
    + split; [ intros [ty Hfind]; discriminate | ].
      intros [[ty [btm [Hsome _]]] | [_ [Hmem _]]]; discriminate.
    + cbn [Typing_env.check_optional_kind_matches
           Typing_env.initial_symbol_type fst].
      destruct (among s (Typing_env.defined_symbols tenv)).
      * change (FlambdaTypeGrammar.kind any_non_null_value) with Value.
        destruct (kind_eqb kind Value) eqn:Hkind.
        -- apply kind_eqb_eq in Hkind. split.
           ++ intros _. right. split; [ reflexivity | ].
              split; [ reflexivity | exact Hkind ].
           ++ intros _. eexists. reflexivity.
        -- split; [ intros [ty Hfind]; discriminate | ].
           intros [[ty [btm [Hsome _]]] | [_ [_ Hk]]]; [ discriminate | ].
           subst kind. discriminate.
      * split; [ intros [ty Hfind]; discriminate | ].
        intros [[ty [btm [Hsome _]]] | [_ [Hmem _]]]; discriminate.
Qed.

(* The same, in a form that [rewrite] takes. *)
Lemma simple_of_kind_name_iff (tenv : Typing_env.t) (n : name)
    (kind : flambda_kind) :
  simple_of_kind tenv (simple_of_name n) kind
  <-> (∃ (ty : flambda_type)
         (binding_time_and_mode : Binding_time.With_name_mode.t),
         Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
         = Some (ty, binding_time_and_mode)
         ∧ FlambdaTypeGrammar.kind ty = kind)
      ∨ (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = None
         ∧ Typing_env.mem None tenv n = true
         ∧ kind = Value).
Proof. exact (simple_of_kind_name tenv n kind). Qed.

Theorem simple_of_kind_unique : simple_of_kind_unique_statement.
Proof.
  intros tenv s kind kind'.
  destruct s as [x | sym | c].
  - change (Var x) with (simple_of_name (Name_var x)).
    rewrite !simple_of_kind_name_iff.
    intros [[ty [btm [Hlookup <-]]] | [Hnone [_ ->]]]
      [[ty' [btm' [Hlookup' <-]]] | [Hnone' [_ ->]]];
      try congruence.
  - change (Symbol sym) with (simple_of_name (Name_symbol sym)).
    rewrite !simple_of_kind_name_iff.
    intros [[ty [btm [Hlookup <-]]] | [Hnone [_ ->]]]
      [[ty' [btm' [Hlookup' <-]]] | [Hnone' [_ ->]]];
      try congruence.
  - simpl. congruence.
Qed.

(* [simple_of_kind] reads the types of the names and the symbols that
   are defined, and nothing else. *)
Lemma simple_of_kind_same (tenv tenv' : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  Typing_env.names_to_types tenv' = Typing_env.names_to_types tenv ->
  Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv ->
  (simple_of_kind tenv' s kind <-> simple_of_kind tenv s kind).
Proof.
  intros Htypes Hsymbols.
  destruct s as [x | sym | c]; [ | | reflexivity ].
  - change (Var x) with (simple_of_name (Name_var x)).
    rewrite !simple_of_kind_name_iff, !mem_var_is_bound, Htypes.
    reflexivity.
  - change (Symbol sym) with (simple_of_name (Name_symbol sym)).
    rewrite !simple_of_kind_name_iff, Htypes.
    change (Typing_env.mem None tenv' (Name_symbol sym))
      with (among sym (Typing_env.defined_symbols tenv')).
    rewrite Hsymbols. reflexivity.
Qed.

Theorem simple_of_kind_kept : simple_of_kind_kept_statement.
Proof.
  intros tenv tenv' s kind Hentries Hsymbols Hvalue.
  assert (Hname :
    ∀ n : name,
      simple_of_kind tenv (simple_of_name n) kind ->
      simple_of_kind tenv' (simple_of_name n) kind).
  { intros n. rewrite !simple_of_kind_name_iff.
    intros [[ty [btm [Hlookup Hkind]]] | [Hnone [Hmem ->]]].
    - destruct (Hentries n ty btm Hlookup) as [ty' [btm' [Hlookup' Hkind']]].
      left. exists ty', btm'. split; [ exact Hlookup' | congruence ].
    - destruct n as [x | sym].
      + rewrite mem_var_is_bound in Hmem. unfold Keyed.is_bound in Hmem.
        rewrite Hnone in Hmem. discriminate.
      + destruct (Keyed.lookup name_eqb (Name_symbol sym)
                    (Typing_env.names_to_types tenv'))
          as [[ty btm] | ] eqn:Hlookup'.
        * left. exists ty, btm. split; [ reflexivity | ].
          exact (Hvalue sym ty btm Hnone Hlookup').
        * right. split; [ reflexivity | ]. split; [ | reflexivity ].
          exact (Hsymbols sym Hmem). }
  destruct s as [x | sym | c]; [ | | exact (fun H => H) ].
  - exact (Hname (Name_var x)).
  - exact (Hname (Name_symbol sym)).
Qed.

(* ======================================================================== *)
(* What the functions do to the kind of a [Simple]                          *)
(* ======================================================================== *)

Theorem make_bottom_simple_of_kind : make_bottom_simple_of_kind_statement.
Proof. intros tenv s kind. apply simple_of_kind_same; reflexivity. Qed.

Theorem increment_scope_simple_of_kind
    : increment_scope_simple_of_kind_statement.
Proof. intros tenv s kind. apply simple_of_kind_same; reflexivity. Qed.

Theorem closure_env_simple_of_kind : closure_env_simple_of_kind_statement.
Proof. intros tenv s kind. apply simple_of_kind_same; reflexivity. Qed.

Theorem add_symbol_definition_simple_of_kind
    : add_symbol_definition_simple_of_kind_statement.
Proof.
  intros tenv sym s kind.
  destruct s as [x | sym' | c].
  - change (Var x) with (simple_of_name (Name_var x)).
    rewrite !simple_of_kind_name_iff. cbn [simple_of_name].
    change (Typing_env.names_to_types
              (Typing_env.add_symbol_definition tenv sym))
      with (Typing_env.names_to_types tenv).
    split; [ intros H; left; exact H | ].
    intros [H | [Heq _]]; [ exact H | discriminate ].
  - change (Symbol sym') with (simple_of_name (Name_symbol sym')).
    rewrite !simple_of_kind_name_iff. cbn [simple_of_name].
    change (Typing_env.names_to_types
              (Typing_env.add_symbol_definition tenv sym))
      with (Typing_env.names_to_types tenv).
    change (Typing_env.mem None (Typing_env.add_symbol_definition tenv sym)
              (Name_symbol sym'))
      with (among sym' (sym :: Typing_env.defined_symbols tenv)).
    change (Typing_env.mem None tenv (Name_symbol sym'))
      with (among sym' (Typing_env.defined_symbols tenv)).
    rewrite among_cons. split.
    + intros [Hentry | [Hnone [Hmem ->]]];
        [ left; left; exact Hentry | ].
      apply orb_true_iff in Hmem. destruct Hmem as [Heq | Hmem].
      * apply Pos.eqb_eq in Heq. subst sym'. right.
        split; [ reflexivity | ]. split; [ reflexivity | exact Hnone ].
      * left. right. split; [ exact Hnone | ].
        split; [ exact Hmem | reflexivity ].
    + intros [[Hentry | [Hnone [Hmem ->]]] | [[= ->] [-> Hnone]]].
      * left. exact Hentry.
      * right. split; [ exact Hnone | ]. split; [ | reflexivity ].
        rewrite Hmem. apply orb_true_r.
      * right. split; [ exact Hnone | ]. split; [ | reflexivity ].
        rewrite Pos.eqb_refl. reflexivity.
  - simpl. split; [ intros H; left; exact H | ].
    intros [H | [Heq _]]; [ exact H | discriminate ].
Qed.

Theorem add_variable_definition_simple_of_kind
    : add_variable_definition_simple_of_kind_statement.
Proof.
  intros tenv var kind_of_var name_mode s kind.
  destruct s as [x | sym | c].
  - change (Var x) with (simple_of_name (Name_var x)).
    rewrite !simple_of_kind_name_iff, add_variable_definition_lookup.
    cbn [simple_of_name].
    destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_var x) (Name_var var))
      as [Heq | Hne].
    + injection Heq as ->. split.
      * intros [[ty [btm [[= <- <-] Hkind]]] | [Hnone _]]; [ | discriminate ].
        left. split; [ reflexivity | ].
        rewrite kind_unknown in Hkind. symmetry. exact Hkind.
      * intros [[_ ->] | [Hne _]]; [ | contradiction ].
        left. eexists. eexists. split; [ reflexivity | apply kind_unknown ].
    + assert (Hmem :
        Typing_env.mem None
          (Typing_env.add_variable_definition tenv var kind_of_var name_mode)
          (Name_var x)
        = Typing_env.mem None tenv (Name_var x)).
      { apply mem_add_variable_definition. congruence. }
      rewrite Hmem. split.
      * intros H. right. split; [ congruence | exact H ].
      * intros [[Heq _] | [_ H]]; [ congruence | exact H ].
  - change (Symbol sym) with (simple_of_name (Name_symbol sym)).
    rewrite !simple_of_kind_name_iff, add_variable_definition_lookup.
    cbn [simple_of_name].
    destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_symbol sym)
                (Name_var var)) as [Heq | _]; [ discriminate | ].
    change (Typing_env.mem None
              (Typing_env.add_variable_definition tenv var kind_of_var
                 name_mode) (Name_symbol sym))
      with (Typing_env.mem None tenv (Name_symbol sym)).
    split.
    + intros H. right. split; [ discriminate | exact H ].
    + intros [[Heq _] | [_ H]]; [ discriminate | exact H ].
  - simpl. split.
    + intros H. right. split; [ discriminate | exact H ].
    + intros [[Heq _] | [_ H]]; [ discriminate | exact H ].
Qed.

Theorem replace_equation_simple_of_kind
    : replace_equation_simple_of_kind_statement.
Proof.
  intros tenv n ty s kind Hkind.
  apply simple_of_kind_name_iff in Hkind.
  assert (Hsymbols :
    ∀ sym : symbol,
      Typing_env.mem None (Typing_env.replace_equation tenv n ty)
        (Name_symbol sym)
      = Typing_env.mem None tenv (Name_symbol sym)) by reflexivity.
  (* The same for a name. *)
  assert (Hname :
    ∀ m : name,
      simple_of_kind (Typing_env.replace_equation tenv n ty)
        (simple_of_name m) kind
      <-> simple_of_kind tenv (simple_of_name m) kind).
  { intros m. rewrite !simple_of_kind_name_iff.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [-> | Hne].
    - rewrite (replace_equation_lookup tenv n ty).
      destruct Hkind
        as [[ty0 [btm0 [Hlookup Hkind0]]] | [Hnone [Hmem Hvalue]]].
      + rewrite Hlookup. destruct n as [x | sym]; simpl.
        * split.
          -- intros [[ty1 [btm1 [[= <- <-] Hk]]] | [Hnone _]];
               [ | discriminate ].
             left. exists ty0, btm0. split; [ reflexivity | congruence ].
          -- intros [[ty1 [btm1 [[= <- <-] Hk]]] | [Hnone _]];
               [ | discriminate ].
             left. eexists. eexists. split; [ reflexivity | congruence ].
        * split.
          -- intros [[ty1 [btm1 [[= <- <-] Hk]]] | [Hnone _]];
               [ | discriminate ].
             left. exists ty0, btm0. split; [ reflexivity | congruence ].
          -- intros [[ty1 [btm1 [[= <- <-] Hk]]] | [Hnone _]];
               [ | discriminate ].
             left. eexists. eexists. split; [ reflexivity | congruence ].
      + destruct n as [x | sym].
        * exfalso. rewrite mem_var_is_bound in Hmem.
          unfold Keyed.is_bound in Hmem. rewrite Hnone in Hmem.
          discriminate.
        * rewrite Hnone. simpl. split.
          -- intros [[ty1 [btm1 [[= <- <-] Hk]]] | [Hnone' _]];
               [ | discriminate ].
             right. split; [ reflexivity | ].
             split; [ exact Hmem | congruence ].
          -- intros [[ty1 [btm1 [Hsome _]]] | [_ [_ ->]]];
               [ discriminate | ].
             left. eexists. eexists. split; [ reflexivity | exact Hvalue ].
    - rewrite (replace_equation_lookup_other tenv n m ty Hne).
      destruct m as [y | sym]; [ | rewrite Hsymbols; reflexivity ].
      rewrite !mem_var_is_bound. unfold Keyed.is_bound.
      rewrite (replace_equation_lookup_other tenv n (Name_var y) ty Hne).
      reflexivity. }
  destruct s as [x | sym | c]; [ | | reflexivity ].
  - exact (Hname (Name_var x)).
  - exact (Hname (Name_symbol sym)).
Qed.

Theorem add_alias_simple_of_kind : add_alias_simple_of_kind_statement.
Proof.
  intros tenv c1 c2 result s kind Hadd.
  destruct (add_alias_frame tenv c1 c2 result Hadd)
    as [_ [Hsymbols [_ [_ [Htypes _]]]]].
  apply simple_of_kind_same; assumption.
Qed.

(* ======================================================================== *)
(* The functions keep the kinds of what is known to be equal                *)
(* ======================================================================== *)

(* Environments with the same aliases in which the simples have the
   same kinds. *)
Lemma aliases_kinded_same (tenv tenv' : Typing_env.t) :
  Typing_env.aliases tenv' = Typing_env.aliases tenv ->
  (∀ (s : simple) (kind : flambda_kind),
    simple_of_kind tenv' s kind -> simple_of_kind tenv s kind) ->
  aliases_kinded tenv ->
  aliases_kinded tenv'.
Proof.
  intros Haliases Hkinds Hkinded e e' kind kind' Haliased He He'.
  rewrite Haliases in Haliased.
  exact (Hkinded e e' kind kind' Haliased (Hkinds e kind He)
           (Hkinds e' kind' He')).
Qed.

Theorem create_aliases_kinded : create_aliases_kinded_statement.
Proof.
  intros e e' kind kind' Haliased He He'.
  assert (Heq : e = e').
  { unfold aliased in Haliased.
    destruct (canonical_cases (Typing_env.aliases Typing_env.create) e)
      as [[_ Hself] | [n [c [_ [Hdemoted _]]]]]; [ | discriminate Hdemoted ].
    destruct (canonical_cases (Typing_env.aliases Typing_env.create) e')
      as [[_ Hself'] | [n [c [_ [Hdemoted _]]]]]; [ | discriminate Hdemoted ].
    congruence. }
  subst e'. exact (simple_of_kind_unique _ e kind kind' He He').
Qed.

Theorem make_bottom_aliases_kinded : make_bottom_aliases_kinded_statement.
Proof.
  intros tenv. apply aliases_kinded_same; [ reflexivity | ].
  intros s kind. apply make_bottom_simple_of_kind.
Qed.

Theorem increment_scope_aliases_kinded
    : increment_scope_aliases_kinded_statement.
Proof.
  intros tenv. apply aliases_kinded_same; [ reflexivity | ].
  intros s kind. apply increment_scope_simple_of_kind.
Qed.

Theorem closure_env_aliases_kinded : closure_env_aliases_kinded_statement.
Proof.
  intros tenv. apply aliases_kinded_same; [ reflexivity | ].
  intros s kind. apply closure_env_simple_of_kind.
Qed.

Theorem add_symbol_definition_aliases_kinded
    : add_symbol_definition_aliases_kinded_statement.
Proof.
  intros tenv sym Hwf Hkinded e e' kind kind' Haliased He He'.
  change (Typing_env.aliases (Typing_env.add_symbol_definition tenv sym))
    with (Typing_env.aliases tenv) in Haliased.
  destruct (simple_equal e e') eqn:Hequal.
  { apply simple_equal_decides_equality in Hequal. subst e'.
    exact (simple_of_kind_unique _ e kind kind' He He'). }
  assert (Hne : e <> e').
  { intros Heq. apply simple_equal_decides_equality in Heq. congruence. }
  (* A simple that is known to be equal to another was of its kind. *)
  assert (Hold :
    ∀ (a b : simple) (k : flambda_kind),
      aliased (Typing_env.aliases tenv) a b ->
      a <> b ->
      simple_of_kind (Typing_env.add_symbol_definition tenv sym) a k ->
      simple_of_kind tenv a k).
  { intros a b k Hab Hab_ne Ha.
    apply add_symbol_definition_simple_of_kind in Ha.
    destruct Ha as [Ha | [-> [-> Hnone]]]; [ exact Ha | ].
    pose proof (aliased_mentioned _ _ _ (Name_symbol sym) Hab Hab_ne eq_refl)
      as Hmention.
    pose proof (well_formed_mentioned_known tenv _ Hwf Hmention) as Hbound.
    apply (simple_of_kind_name_iff tenv (Name_symbol sym) Value). right.
    split; [ exact Hnone | ]. split; [ exact Hbound | reflexivity ]. }
  apply (Hkinded e e' kind kind' Haliased).
  - exact (Hold e e' kind Haliased Hne He).
  - apply (Hold e' e kind'); [ | congruence | exact He' ].
    unfold aliased in *. congruence.
Qed.

Theorem add_variable_definition_aliases_kinded
    : add_variable_definition_aliases_kinded_statement.
Proof.
  intros tenv var kind_of_var name_mode Hwf Hfresh Hkinded e e' kind kind'
    Haliased He He'.
  change (Typing_env.aliases
            (Typing_env.add_variable_definition tenv var kind_of_var
               name_mode))
    with (Typing_env.aliases tenv) in Haliased.
  destruct (simple_equal e e') eqn:Hequal.
  { apply simple_equal_decides_equality in Hequal. subst e'.
    exact (simple_of_kind_unique _ e kind kind' He He'). }
  assert (Hne : e <> e').
  { intros Heq. apply simple_equal_decides_equality in Heq. congruence. }
  assert (Hold :
    ∀ (a b : simple) (k : flambda_kind),
      aliased (Typing_env.aliases tenv) a b ->
      a <> b ->
      simple_of_kind
        (Typing_env.add_variable_definition tenv var kind_of_var name_mode) a
        k ->
      simple_of_kind tenv a k).
  { intros a b k Hab Hab_ne Ha.
    apply add_variable_definition_simple_of_kind in Ha.
    destruct Ha as [[-> _] | [_ Ha]]; [ exfalso | exact Ha ].
    pose proof (aliased_mentioned _ _ _ (Name_var var) Hab Hab_ne eq_refl)
      as Hmention.
    pose proof (well_formed_mentioned_known tenv _ Hwf Hmention) as Hbound.
    congruence. }
  apply (Hkinded e e' kind kind' Haliased).
  - exact (Hold e e' kind Haliased Hne He).
  - apply (Hold e' e kind'); [ | congruence | exact He' ].
    unfold aliased in *. congruence.
Qed.

Theorem replace_equation_aliases_kinded
    : replace_equation_aliases_kinded_statement.
Proof.
  intros tenv n ty Hkind.
  apply aliases_kinded_same; [ apply replace_equation_aliases | ].
  intros s kind.
  apply (replace_equation_simple_of_kind tenv n ty s kind Hkind).
Qed.

Theorem add_alias_aliases_kinded : add_alias_aliases_kinded_statement.
Proof.
  intros tenv c1 c2 result kind Hconsistent Hc1 Hc2 Hkind1 Hkind2 Hadd
    Hkinded e e' k k' Haliased He He'.
  apply (add_alias_simple_of_kind tenv c1 c2 result e k Hadd) in He.
  apply (add_alias_simple_of_kind tenv c1 c2 result e' k' Hadd) in He'.
  destruct (add_alias_frame tenv c1 c2 result Hadd)
    as [[added [Hadded [Haliases _]]] _].
  rewrite Haliases in Haliased.
  apply (add_aliased _ _ _ _ c1 c2 added e e' Hconsistent Hc1 Hc2 Hadded)
    in Haliased.
  destruct Haliased as [Hold | [[H1 H2] | [H2 H1]]].
  - exact (Hkinded e e' k k' Hold He He').
  - rewrite (Hkinded e c1 k kind H1 He Hkind1).
    exact (eq_sym (Hkinded e' c2 k' kind H2 He' Hkind2)).
  - rewrite (Hkinded e c2 k kind H2 He Hkind2).
    exact (eq_sym (Hkinded e' c1 k' kind H1 He' Hkind1)).
Qed.

(* ======================================================================== *)
(* Constants                                                                *)
(* ======================================================================== *)

Theorem aliases_kinded_constants_of_kind
    : aliases_kinded_constants_of_kind_statement.
Proof.
  intros tenv Hconsistent Hkinded s kind c Hs.
  unfold Typing_env.get_canonical_simple_exn.
  destruct (option_map Binding_time.With_name_mode.name_mode
              (Typing_env.binding_time_and_mode_of_simple tenv s))
    as [name_mode_simple | ]; [ | discriminate ].
  intros [= Hcanonical].
  pose proof (canonical_at_mode_aliased _ _ _ _ _ _ _ _ _ Hconsistent
                Hcanonical) as Haliased.
  exact (Hkinded (Const c) s _ kind Haliased eq_refl Hs).
Qed.

(* ======================================================================== *)
(* What a kinded environment gives                                          *)
(* ======================================================================== *)

Theorem type_kinded_kept : type_kinded_kept_statement.
Proof.
  intros tenv tenv' ty Hkept Hkinded s kind Hin.
  exact (Hkept s kind (Hkinded s kind Hin)).
Qed.

Theorem find_kinded : find_kinded_statement.
Proof.
  intros tenv n kind ty Hkinded Hfind.
  destruct (find_cases tenv n kind ty Hfind)
    as [[binding_time_and_mode Hlookup] | [[k ->] | ->]].
  - exact (Hkinded n ty binding_time_and_mode Hlookup).
  - intros s kind' Hin. rewrite (aliases_with_kinds_bottom k) in Hin.
    destruct Hin.
  - intros s kind' Hin. destruct Hin.
Qed.

Theorem symbol_of_kind_value : symbol_of_kind_value_statement.
Proof.
  intros tenv s kind Hsymbols Hs.
  apply (simple_of_kind_name_iff tenv (Name_symbol s) kind) in Hs.
  destruct Hs as [[ty [btm [Hlookup <-]]] | [_ [_ ->]]]; [ | reflexivity ].
  exact (Hsymbols s ty btm Hlookup).
Qed.

Theorem canonical_ignoring_name_mode_kinded
    : canonical_ignoring_name_mode_kinded_statement.
Proof.
  intros tenv s kind Hwf Hkinded Hs.
  pose proof (proj1 (typing_env_well_formed_iff tenv) Hwf) as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  assert (Hc :
    Typing_env.get_canonical_simple_ignoring_name_mode tenv s
    = canonical_element_of (Typing_env.aliases tenv) s).
  { destruct s as [x | sym | k]; reflexivity. }
  rewrite Hc.
  set (c := canonical_element_of (Typing_env.aliases tenv) s) in *.
  destruct (simple_equal c s) eqn:Hequal.
  { apply simple_equal_decides_equality in Hequal. rewrite Hequal.
    exact Hs. }
  assert (Hne : c <> s).
  { intros Heq. apply simple_equal_decides_equality in Heq. congruence. }
  assert (Haliased : aliased (Typing_env.aliases tenv) c s).
  { unfold aliased, c. apply canonical_element_idempotent.
    exact Hconsistent. }
  (* The canonical element is of some kind. *)
  assert (Hhas : ∃ kind' : flambda_kind, simple_of_kind tenv c kind').
  { destruct c as [x | sym | k] eqn:Hform.
    - pose proof (well_formed_mentioned_known tenv _ Hwf
                    (aliased_mentioned _ _ _ (Name_var x) Haliased Hne
                       eq_refl)) as Hbound.
      apply mem_var_lookup in Hbound. destruct Hbound as [ty [btm Hlookup]].
      exists (FlambdaTypeGrammar.kind ty).
      apply (simple_of_kind_name_iff tenv (Name_var x)). left.
      exists ty, btm. split; [ exact Hlookup | reflexivity ].
    - pose proof (well_formed_mentioned_known tenv _ Hwf
                    (aliased_mentioned _ _ _ (Name_symbol sym) Haliased Hne
                       eq_refl)) as Hbound.
      destruct (Keyed.lookup name_eqb (Name_symbol sym)
                  (Typing_env.names_to_types tenv))
        as [[ty btm] | ] eqn:Hlookup.
      + exists (FlambdaTypeGrammar.kind ty).
        apply (simple_of_kind_name_iff tenv (Name_symbol sym)). left.
        exists ty, btm. split; [ exact Hlookup | reflexivity ].
      + exists Value.
        apply (simple_of_kind_name_iff tenv (Name_symbol sym)). right.
        split; [ exact Hlookup | ]. split; [ exact Hbound | reflexivity ].
    - exists (FlambdaTypeGrammar.kind (type_for_const k)). reflexivity. }
  destruct Hhas as [kind' Hkind'].
  rewrite (Hkinded c s kind' kind Haliased Hkind' Hs) in Hkind'.
  exact Hkind'.
Qed.

Theorem typing_env_kinded_constants_of_kind
    : typing_env_kinded_constants_of_kind_statement.
Proof.
  intros tenv Hconsistent [_ [Haliases _]].
  exact (aliases_kinded_constants_of_kind tenv Hconsistent Haliases).
Qed.

(* ======================================================================== *)
(* The functions keep an environment kinded                                 *)
(* ======================================================================== *)

(* Environments with the same types of the names in which the simples
   keep their kinds. *)
Lemma types_kinded_same (tenv tenv' : Typing_env.t) :
  Typing_env.names_to_types tenv' = Typing_env.names_to_types tenv ->
  (∀ (s : simple) (kind : flambda_kind),
    simple_of_kind tenv s kind -> simple_of_kind tenv' s kind) ->
  types_kinded tenv ->
  types_kinded tenv'.
Proof.
  intros Htypes Hkept Hkinded n ty btm Hlookup. rewrite Htypes in Hlookup.
  exact (type_kinded_kept tenv tenv' ty Hkept (Hkinded n ty btm Hlookup)).
Qed.

Lemma symbols_of_kind_value_same (tenv tenv' : Typing_env.t) :
  Typing_env.names_to_types tenv' = Typing_env.names_to_types tenv ->
  symbols_of_kind_value tenv ->
  symbols_of_kind_value tenv'.
Proof.
  intros Htypes Hsymbols s ty btm Hlookup. rewrite Htypes in Hlookup.
  exact (Hsymbols s ty btm Hlookup).
Qed.

Theorem create_kinded : create_kinded_statement.
Proof.
  split; [ | split; [ exact create_aliases_kinded | ] ].
  - intros n ty btm Hlookup. discriminate.
  - intros s ty btm Hlookup. discriminate.
Qed.

Theorem make_bottom_kinded : make_bottom_kinded_statement.
Proof.
  intros tenv [Htypes [Haliases Hsymbols]].
  split; [ | split ].
  - apply (types_kinded_same tenv); [ reflexivity | | exact Htypes ].
    intros s kind. apply make_bottom_simple_of_kind.
  - exact (make_bottom_aliases_kinded tenv Haliases).
  - exact Hsymbols.
Qed.

Theorem increment_scope_kinded : increment_scope_kinded_statement.
Proof.
  intros tenv [Htypes [Haliases Hsymbols]].
  split; [ | split ].
  - apply (types_kinded_same tenv); [ reflexivity | | exact Htypes ].
    intros s kind. apply increment_scope_simple_of_kind.
  - exact (increment_scope_aliases_kinded tenv Haliases).
  - exact Hsymbols.
Qed.

Theorem closure_env_kinded : closure_env_kinded_statement.
Proof.
  intros tenv [Htypes [Haliases Hsymbols]].
  split; [ | split ].
  - apply (types_kinded_same tenv); [ reflexivity | | exact Htypes ].
    intros s kind. apply closure_env_simple_of_kind.
  - exact (closure_env_aliases_kinded tenv Haliases).
  - exact Hsymbols.
Qed.

Theorem add_symbol_definition_kinded : add_symbol_definition_kinded_statement.
Proof.
  intros tenv sym Hwf [Htypes [Haliases Hsymbols]].
  split; [ | split ].
  - apply (types_kinded_same tenv); [ reflexivity | | exact Htypes ].
    intros s kind Hs. apply add_symbol_definition_simple_of_kind. left.
    exact Hs.
  - exact (add_symbol_definition_aliases_kinded tenv sym Hwf Haliases).
  - exact Hsymbols.
Qed.

Theorem add_variable_definition_kinded
    : add_variable_definition_kinded_statement.
Proof.
  intros tenv var kind_of_var name_mode Hwf Hfresh
    [Htypes [Haliases Hsymbols]].
  (* A simple that was of a kind is not the variable. *)
  assert (Hkept :
    ∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind ->
      simple_of_kind
        (Typing_env.add_variable_definition tenv var kind_of_var name_mode) s
        kind).
  { intros s kind Hs. apply add_variable_definition_simple_of_kind. right.
    split; [ | exact Hs ]. intros ->.
    apply (simple_of_kind_name_iff tenv (Name_var var) kind) in Hs.
    pose proof (mem_fresh_lookup tenv var Hfresh) as Hnone.
    destruct Hs as [[ty [btm [Hlookup _]]] | [_ [Hmem _]]]; congruence. }
  split; [ | split ].
  - intros n ty btm Hlookup. rewrite add_variable_definition_lookup in Hlookup.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq n (Name_var var))
      as [-> | Hne].
    + injection Hlookup as <- <-. intros s kind Hin.
      rewrite (aliases_with_kinds_unknown kind_of_var) in Hin. destruct Hin.
    + exact (type_kinded_kept tenv _ ty Hkept (Htypes n ty btm Hlookup)).
  - exact (add_variable_definition_aliases_kinded tenv var kind_of_var
             name_mode Hwf Hfresh Haliases).
  - intros s ty btm Hlookup. rewrite add_variable_definition_lookup in Hlookup.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_symbol s)
                (Name_var var)) as [Heq | _]; [ discriminate | ].
    exact (Hsymbols s ty btm Hlookup).
Qed.

Theorem replace_equation_kinded : replace_equation_kinded_statement.
Proof.
  intros tenv n ty Hkind Hty [Htypes [Haliases Hsymbols]].
  assert (Hkept :
    ∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind ->
      simple_of_kind (Typing_env.replace_equation tenv n ty) s kind).
  { intros s kind. apply (replace_equation_simple_of_kind tenv n ty s kind
                            Hkind). }
  split; [ | split ].
  - intros m ty1 btm1 Hlookup.
    destruct (replace_equation_lookup_cases tenv n ty m ty1 btm1 Hlookup)
      as [[_ ->] | Hold].
    + exact (type_kinded_kept tenv _ ty Hkept Hty).
    + exact (type_kinded_kept tenv _ ty1 Hkept (Htypes m ty1 btm1 Hold)).
  - exact (replace_equation_aliases_kinded tenv n ty Hkind Haliases).
  - intros s ty1 btm1 Hlookup.
    destruct (replace_equation_lookup_cases tenv n ty (Name_symbol s) ty1
                btm1 Hlookup) as [[<- ->] | Hold].
    + exact (symbol_of_kind_value tenv s _ Hsymbols Hkind).
    + exact (Hsymbols s ty1 btm1 Hold).
Qed.

Theorem add_alias_kinded : add_alias_kinded_statement.
Proof.
  intros tenv c1 c2 result kind Hconsistent Hc1 Hc2 Hkind1 Hkind2 Hadd
    [Htypes [Haliases Hsymbols]].
  destruct (add_alias_frame tenv c1 c2 result Hadd)
    as [_ [_ [_ [_ [Hsame _]]]]].
  split; [ | split ].
  - apply (types_kinded_same tenv); [ exact Hsame | | exact Htypes ].
    intros s k. apply (add_alias_simple_of_kind tenv c1 c2 result s k Hadd).
  - exact (add_alias_aliases_kinded tenv c1 c2 result kind Hconsistent Hc1
             Hc2 Hkind1 Hkind2 Hadd Haliases).
  - exact (symbols_of_kind_value_same tenv _ Hsame Hsymbols).
Qed.

(* ======================================================================== *)
(* The equations inside types                                               *)
(* ======================================================================== *)

Theorem type_equations_kinded_kept : type_equations_kinded_kept_statement.
Proof.
  intros tenv tenv' ty Hkept Hkinded n kind Hin.
  exact (Hkept _ kind (Hkinded n kind Hin)).
Qed.

Theorem find_equations_kinded : find_equations_kinded_statement.
Proof.
  intros tenv n kind ty Hkinded Hfind.
  destruct (find_cases tenv n kind ty Hfind)
    as [[binding_time_and_mode Hlookup] | [[k ->] | ->]].
  - exact (Hkinded n ty binding_time_and_mode Hlookup).
  - intros m kind' Hin.
    rewrite (proj1 (proj2 equation_names_none) k) in Hin. destruct Hin.
  - intros m kind' Hin. destruct Hin.
Qed.

Lemma types_equations_kinded_same (tenv tenv' : Typing_env.t) :
  Typing_env.names_to_types tenv' = Typing_env.names_to_types tenv ->
  (∀ (s : simple) (kind : flambda_kind),
    simple_of_kind tenv s kind -> simple_of_kind tenv' s kind) ->
  types_equations_kinded tenv ->
  types_equations_kinded tenv'.
Proof.
  intros Htypes Hkept Hkinded n ty btm Hlookup. rewrite Htypes in Hlookup.
  exact (type_equations_kinded_kept tenv tenv' ty Hkept
           (Hkinded n ty btm Hlookup)).
Qed.

Theorem create_equations_kinded : create_equations_kinded_statement.
Proof. intros n ty btm Hlookup. discriminate. Qed.

Theorem make_bottom_equations_kinded
    : make_bottom_equations_kinded_statement.
Proof.
  intros tenv Hequations.
  apply (types_equations_kinded_same tenv);
    [ reflexivity | | exact Hequations ].
  intros s kind. apply make_bottom_simple_of_kind.
Qed.

Theorem increment_scope_equations_kinded
    : increment_scope_equations_kinded_statement.
Proof.
  intros tenv Hequations.
  apply (types_equations_kinded_same tenv);
    [ reflexivity | | exact Hequations ].
  intros s kind. apply increment_scope_simple_of_kind.
Qed.

Theorem closure_env_equations_kinded
    : closure_env_equations_kinded_statement.
Proof.
  intros tenv Hequations.
  apply (types_equations_kinded_same tenv);
    [ reflexivity | | exact Hequations ].
  intros s kind. apply closure_env_simple_of_kind.
Qed.

Theorem add_symbol_definition_equations_kinded
    : add_symbol_definition_equations_kinded_statement.
Proof.
  intros tenv sym Hequations.
  apply (types_equations_kinded_same tenv);
    [ reflexivity | | exact Hequations ].
  intros s kind Hs. apply add_symbol_definition_simple_of_kind. left.
  exact Hs.
Qed.

Theorem add_variable_definition_equations_kinded
    : add_variable_definition_equations_kinded_statement.
Proof.
  intros tenv var kind_of_var name_mode Hfresh Hequations.
  (* A simple that was of a kind is not the variable. *)
  assert (Hkept :
    ∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind ->
      simple_of_kind
        (Typing_env.add_variable_definition tenv var kind_of_var name_mode) s
        kind).
  { intros s kind Hs. apply add_variable_definition_simple_of_kind. right.
    split; [ | exact Hs ]. intros ->.
    apply (simple_of_kind_name_iff tenv (Name_var var) kind) in Hs.
    pose proof (mem_fresh_lookup tenv var Hfresh) as Hnone.
    destruct Hs as [[ty [btm [Hlookup _]]] | [_ [Hmem _]]]; congruence. }
  intros n ty btm Hlookup. rewrite add_variable_definition_lookup in Hlookup.
  destruct (keyed_eqb_spec name_eqb name_eqb_eq n (Name_var var))
    as [-> | Hne].
  - injection Hlookup as <- <-. intros m kind Hin.
    rewrite (proj1 equation_names_none kind_of_var) in Hin. destruct Hin.
  - exact (type_equations_kinded_kept tenv _ ty Hkept
             (Hequations n ty btm Hlookup)).
Qed.

Theorem replace_equation_equations_kinded
    : replace_equation_equations_kinded_statement.
Proof.
  intros tenv n ty Hkind Hty Hequations.
  assert (Hkept :
    ∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind ->
      simple_of_kind (Typing_env.replace_equation tenv n ty) s kind).
  { intros s kind. apply (replace_equation_simple_of_kind tenv n ty s kind
                            Hkind). }
  intros m ty1 btm1 Hlookup.
  destruct (replace_equation_lookup_cases tenv n ty m ty1 btm1 Hlookup)
    as [[_ ->] | Hold].
  - exact (type_equations_kinded_kept tenv _ ty Hkept Hty).
  - exact (type_equations_kinded_kept tenv _ ty1 Hkept
             (Hequations m ty1 btm1 Hold)).
Qed.

Theorem add_alias_equations_kinded : add_alias_equations_kinded_statement.
Proof.
  intros tenv c1 c2 result Hadd Hequations.
  destruct (add_alias_frame tenv c1 c2 result Hadd)
    as [_ [_ [_ [_ [Hsame _]]]]].
  apply (types_equations_kinded_same tenv);
    [ exact Hsame | | exact Hequations ].
  intros s k. apply (add_alias_simple_of_kind tenv c1 c2 result s k Hadd).
Qed.
