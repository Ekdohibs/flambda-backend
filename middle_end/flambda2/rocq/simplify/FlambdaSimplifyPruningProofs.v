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


(* Proofs of the statements of FlambdaSimplifyPruningSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaContextWhereSpec
  FlambdaContextWhereProofs FlambdaUnusedStaticsSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaSinkContsProofs FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaEnvAliasesInvariantSpec FlambdaEnvAliasesInvariantProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaMeet FlambdaMeetSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPrimitivesProofs
  FlambdaSimplifyApplySpec FlambdaSimplifyApplyProofs
  FlambdaSimplifyStaticsSpec FlambdaSimplifyStaticsProofs
  FlambdaSimplifyCodeEnvSpec FlambdaSimplifyCodeEnvProofs
  FlambdaSimplifyHandlersSpec FlambdaSimplifyHandlersProofs
  FlambdaSimplifyPassSpec FlambdaSimplifyUsesProofs
  FlambdaSimplifyPassProofs FlambdaSimplifyPruningSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec
  FlambdaMeetExamplesProofs FlambdaSimplifySwitchSpec
  FlambdaSimplifySwitchProofs.
Import ListNotations.

Local Open Scope Z_scope.

(* [code_env_in] is the function of FlambdaSimplifyCodeEnvSpec.v. The
   theorem of that name in FlambdaSimplifyCodeEnvProofs.v is not used
   here. *)
Local Abbreviation code_env_in :=
  FlambdaSimplifyCodeEnvSpec.code_env_in.

(* ======================================================================== *)
(* The states of the run of a unit whose sets are well formed               *)
(* ======================================================================== *)

Theorem run_valid_kept : run_valid_kept_statement.
Proof.
  intros s s' [Hsets Hvalid] Hstep. split.
  - exact (sets_inv_step s s' Hsets Hstep).
  - pose proof (state_valid_step s Hsets Hvalid) as H.
    rewrite Hstep in H. exact H.
Qed.

Theorem run_valid_initial : run_valid_initial_statement.
Proof.
  intros e Hsets. split; [ | exact (initial_state_valid e) ].
  refine (conj Hsets (conj I (conj _ _))).
  - intros id c H. discriminate H.
  - intros sym slot symbols decls slot_values H. discriminate H.
Qed.

Theorem run_valid_statics : run_valid_statics_statement.
Proof. intros s [_ (_ & _ & Hstatics & _)]. exact Hstatics. Qed.

Theorem run_valid_expr : run_valid_expr_statement.
Proof.
  intros e e' vars conts statics hp rd Hsets [(_ & C & S & B) Hvalid].
  exact (conj (conj Hsets (conj C (conj S B))) Hvalid).
Qed.

(* ======================================================================== *)
(* Refinement from those states                                             *)
(* ======================================================================== *)

(* What [run_valid] says of a state apart from its expression. *)
Definition rest_valid (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    : Prop :=
  run_valid (State Invalid vars conts statics hp rd).

Lemma run_valid_rest (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  run_valid (State e vars conts statics hp rd)
    -> rest_valid vars conts statics hp rd.
Proof. apply run_valid_expr. reflexivity. Qed.

Lemma rest_valid_run (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  sets_well_formed e = true ->
  rest_valid vars conts statics hp rd ->
  run_valid (State e vars conts statics hp rd).
Proof. intros Hsets. apply run_valid_expr. exact Hsets. Qed.

(* A step from a state whose expression has well-formed sets. *)
Lemma rest_valid_step (e e' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  sets_well_formed e = true ->
  rest_valid vars conts statics hp rd ->
  step (State e vars conts statics hp rd)
    = Step (State e' vars' conts' statics' hp rd) ->
  rest_valid vars' conts' statics' hp rd.
Proof.
  intros Hsets Hvalid Hstep. apply (run_valid_rest e').
  exact
    (run_valid_kept _ _ (rest_valid_run e vars conts statics _ rd_ Hsets Hvalid)
           Hstep).
Qed.

(* Refinement from the states that have [run_valid] but for the test of
   the expression: the form in which two steps make one with nothing
   asked of the term between. *)
Definition refines_valid (tenv : Typing_env.t) (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    rest_valid vars conts statics hp rd ->
    env_satisfies_strictly tenv statics vars ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

Lemma refines_valid_from (tenv : Typing_env.t) (source target : expr) :
  refines_valid tenv source target ->
  expr_refines_from_valid tenv source target.
Proof.
  intros H vars conts statics hp rd Hvalid. apply H.
  exact (run_valid_rest source vars conts statics _ rd_ Hvalid).
Qed.

Lemma from_valid_refines_valid (tenv : Typing_env.t) (source target : expr) :
  sets_well_formed source = true ->
  expr_refines_from_valid tenv source target ->
  refines_valid tenv source target.
Proof.
  intros Hsets H vars conts statics hp rd Hvalid. apply H.
  exact (rest_valid_run source vars conts statics _ rd_ Hsets Hvalid).
Qed.

Lemma refines_valid_trans (tenv : Typing_env.t) (e e' e'' : expr) :
  refines_valid tenv e e' -> refines_valid tenv e' e'' ->
  refines_valid tenv e e''.
Proof.
  intros H H' vars conts statics hp rd Hvalid Hsat.
  exact (refines_trans _ _ _ (H vars conts statics hp rd_ Hvalid Hsat)
           (H' vars conts statics hp rd_ Hvalid Hsat)).
Qed.

Lemma refines_under_valid (tenv : Typing_env.t) (e e' : expr) :
  expr_refines_under tenv e e' -> refines_valid tenv e e'.
Proof.
  intros H vars conts statics hp rd _ Hsat. exact
    (H vars conts statics hp rd_ Hsat).
Qed.

Lemma refines_valid_of_under_valid (tenv : Typing_env.t) (e e' : expr) :
  FlambdaSimplifySwitchSpec.expr_refines_under_valid tenv e e' ->
  refines_valid tenv e e'.
Proof.
  intros H vars conts statics hp rd Hvalid Hsat. apply H; [ | exact Hsat ].
  exact (run_valid_statics _ Hvalid).
Qed.

Theorem from_valid_weaken : from_valid_weaken_statement.
Proof.
  intros tenv e e'. split; intros H.
  - apply refines_valid_from, refines_under_valid, H.
  - apply refines_valid_from, refines_valid_of_under_valid, H.
Qed.

Theorem expr_refines_from_valid_trans
    : expr_refines_from_valid_trans_statement.
Proof.
  intros tenv e e' e'' Hsets H H' vars conts statics hp rd Hvalid Hsat.
  refine (refines_trans _ _ _ (H vars conts statics hp rd_ Hvalid Hsat) _).
  apply H'; [ | exact Hsat ].
  exact (run_valid_expr e e' vars conts statics _ rd_ Hsets Hvalid).
Qed.

Theorem from_valid_program : from_valid_program_statement.
Proof.
  intros e e' Hsets H. apply H; [ exact (run_valid_initial e Hsets) | ].
  apply create_satisfied.
Qed.

(* ======================================================================== *)
(* The sets of static bindings whose code is rewritten                      *)
(* ======================================================================== *)

(* A binding without the body of its code. The tests of the sets of a
   group that are not about the bodies of code read no more of it. *)
Definition forget_code_body (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata _) =>
    Bind_code id (Code params my_closure ret exn metadata Invalid)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

Lemma forallb_map_ext {A B : Type} (f : B -> bool) (g : A -> bool)
    (h : A -> B) (l : list A) :
  (∀ a : A, f (h a) = g a) -> forallb f (map h l) = forallb g l.
Proof.
  intros Hext. induction l as [ | a l IH]; [ reflexivity | ].
  cbn [map forallb]. rewrite Hext, IH. reflexivity.
Qed.

Lemma existsb_map_ext {A B : Type} (f : B -> bool) (g : A -> bool)
    (h : A -> B) (l : list A) :
  (∀ a : A, f (h a) = g a) -> existsb f (map h l) = existsb g l.
Proof.
  intros Hext. induction l as [ | a l IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite Hext, IH. reflexivity.
Qed.

Lemma forallb_same {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, f a = g a) -> forallb f l = forallb g l.
Proof.
  intros Hext. rewrite <- (map_id l) at 1. apply forallb_map_ext. exact Hext.
Qed.

Lemma static_set_whole_forget (bindings : list static_binding)
    (binding : static_binding) :
  static_set_whole (map forget_code_body bindings) (forget_code_body binding)
  = static_set_whole bindings binding.
Proof.
  destruct binding as [id [params my_closure ret exn metadata body] | id
                      | s const]; try reflexivity.
  destruct const; try reflexivity. cbn [forget_code_body static_set_whole].
  f_equal. apply forallb_same. intros [slot' s']. apply existsb_map_ext.
  intros [id' [params my_closure ret exn metadata body] | id' | s'' const'];
    reflexivity.
Qed.

Lemma static_sets_code_disjoint_forget (bindings : list static_binding) :
  static_sets_code_disjoint (map forget_code_body bindings)
  = static_sets_code_disjoint bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map static_sets_code_disjoint]. rewrite IH. f_equal.
  apply forallb_map_ext. intros binding'.
  destruct binding as [id [params my_closure ret exn metadata body] | id
                      | s const];
    destruct binding' as [id' [params' my_closure' ret' exn' metadata' body']
                         | id' | s' const'];
    reflexivity.
Qed.

(* Two groups that are the same but for the bodies of their code pass
   the tests that are not about those bodies together. *)
Lemma let_static_sets_alike (bindings bindings' : list static_binding)
    (body body' : expr) :
  map forget_code_body bindings' = map forget_code_body bindings ->
  forallb binding_sets_well_formed bindings' = true ->
  sets_well_formed body' = true ->
  sets_well_formed (Let_static bindings body) = true ->
  sets_well_formed (Let_static bindings' body') = true.
Proof.
  intros Halike Hcode Hbody. cbn [sets_well_formed].
  rewrite !andb_true_iff. intros [[[Hwhole Hdisjoint] _] _].
  refine (conj (conj (conj _ _) Hcode) Hbody).
  - rewrite <- Hwhole.
    rewrite <- (forallb_map_ext
                  (static_set_whole (map forget_code_body bindings'))
                  (static_set_whole bindings') forget_code_body bindings'
                  (static_set_whole_forget bindings')).
    rewrite <- (forallb_map_ext
                  (static_set_whole (map forget_code_body bindings))
                  (static_set_whole bindings) forget_code_body bindings
                  (static_set_whole_forget bindings)).
    rewrite Halike. reflexivity.
  - rewrite <- Hdisjoint, <- static_sets_code_disjoint_forget, Halike.
    apply static_sets_code_disjoint_forget.
Qed.

(* ======================================================================== *)
(* Where a meet may be made                                                 *)
(* ======================================================================== *)

Theorem types_fit_b_fits : types_fit_b_fits_statement.
Proof.
  intros tenv Hinvariant Haliases Hfit.
  assert (Hall :
    ∀ (n : name) (ty : flambda_type) (btm : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, btm) ->
      type_well_formed ty = true
      ∧ type_kinded tenv ty
      ∧ type_equations_kinded tenv ty
      ∧ is_obviously_bottom ty = false
      ∧ (∀ s : symbol, n = Name_symbol s -> kind ty = Value)).
  { intros n ty btm Hlookup.
    pose proof (proj1 (for_all_names_b_iff
                         (fun n entry => type_fits_b tenv n entry = true)
                         (type_fits_b tenv) _ (fun _ _ _ => iff_refl _))
                  Hfit n (ty, btm) Hlookup) as Htest.
    unfold type_fits_b in Htest. cbn [fst] in Htest.
    rewrite !andb_true_iff in Htest.
    destruct Htest as [[[[Hwf Halias] Hequation] Hnot_bottom] Hsymbol].
    split; [ exact Hwf | ].
    split.
    { intros s k Hin. rewrite forallb_forall in Halias.
      exact (simple_of_kind_b_sound tenv s k (Halias (s, k) Hin)). }
    split.
    { intros m k Hin. rewrite forallb_forall in Hequation.
      exact (simple_of_kind_b_sound tenv _ k (Hequation (m, k) Hin)). }
    split; [ apply negb_true_iff; exact Hnot_bottom | ].
    intros s ->. apply kind_eqb_eq. exact Hsymbol. }
  split; [ exact Hinvariant | ].
  split.
  { intros n ty btm Hlookup. exact (proj1 (Hall n ty btm Hlookup)). }
  split.
  { split; [ | split; [ exact Haliases | ] ].
    - intros n ty btm Hlookup. exact (proj1 (proj2 (Hall n ty btm Hlookup))).
    - intros s ty btm Hlookup.
      exact (proj2 (proj2 (proj2 (proj2 (Hall _ ty btm Hlookup)))) s
               eq_refl). }
  split.
  { intros n ty btm Hlookup.
    exact (proj1 (proj2 (proj2 (Hall n ty btm Hlookup)))). }
  intros _ n ty btm Hlookup.
  exact (proj1 (proj2 (proj2 (proj2 (Hall n ty btm Hlookup))))).
Qed.

(* The two conditions that have no test. *)
Definition carried (tenv : Typing_env.t) : Prop :=
  typing_env_invariant tenv ∧ aliases_kinded tenv.

Lemma add_variable_definition_carried (tenv : Typing_env.t) (x : variable)
    (kind : flambda_kind) :
  carried tenv ->
  Typing_env.mem None tenv (Name_var x) = false ->
  carried (Typing_env.add_variable_definition tenv x kind Name_mode.normal).
Proof.
  intros [Hinvariant Haliases] Hfresh. split.
  - exact (add_variable_definition_invariant tenv x kind Name_mode.normal
             Hinvariant Hfresh).
  - exact (add_variable_definition_aliases_kinded tenv x kind
             Name_mode.normal (proj1 Hinvariant) Hfresh Haliases).
Qed.

Lemma add_symbol_definitions_carried (tenv : Typing_env.t)
    (syms : list symbol) :
  carried tenv -> carried (Typing_env.add_symbol_definitions tenv syms).
Proof.
  intros Hcarried. induction syms as [ | sym syms IH].
  - destruct tenv. exact Hcarried.
  - destruct IH as [Hinvariant Haliases]. split.
    + exact (add_symbol_definition_invariant
               (Typing_env.add_symbol_definitions tenv syms) sym Hinvariant).
    + exact (add_symbol_definition_aliases_kinded
               (Typing_env.add_symbol_definitions tenv syms) sym
               (proj1 Hinvariant) Haliases).
Qed.

Lemma add_symbol_definition_carried (tenv : Typing_env.t) (sym : symbol) :
  carried tenv -> carried (Typing_env.add_symbol_definition tenv sym).
Proof.
  intros [Hinvariant Haliases]. split.
  - exact (add_symbol_definition_invariant tenv sym Hinvariant).
  - exact (add_symbol_definition_aliases_kinded tenv sym (proj1 Hinvariant)
             Haliases).
Qed.

(* A name that is given a type, where the test passes. *)
Lemma equation_keeps_fit_carried (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  carried tenv ->
  equation_keeps_fit tenv n ty = true ->
  carried (Typing_env.replace_equation tenv n ty).
Proof.
  intros [Hinvariant Haliases] Hkeeps. unfold equation_keeps_fit in Hkeeps.
  rewrite !andb_true_iff in Hkeeps.
  destruct Hkeeps as [[[[[Hmem Hfree] Hkind] Halias] Hunknown] Hof_kind].
  split.
  - apply (replace_equation_invariant tenv n ty Hinvariant Hmem).
    + intros y Hin. rewrite forallb_forall in Hfree. exact (Hfree y Hin).
    + intros ty' btm Hlookup. rewrite Hlookup in Hkind.
      apply kind_eqb_eq. exact Hkind.
    + intros alias Halias' Hcanonical. rewrite Halias' in Halias.
      apply is_canonical_b_iff in Hcanonical. rewrite Hcanonical in Halias.
      discriminate.
    + apply orb_true_iff in Hunknown. destruct Hunknown as [H | H].
      * left. apply negb_true_iff. exact H.
      * right.
        match goal with
        | |- ?found = None =>
          destruct found eqn:Hlookup;
            [ try rewrite Hlookup in H; cbn in H; discriminate H
            | reflexivity ]
        end.
  - exact (replace_equation_aliases_kinded tenv n ty
             (simple_of_kind_b_sound tenv _ _ Hof_kind) Haliases).
Qed.

Lemma block_like_bindings_keep_fit_carried (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    carried tenv ->
    block_like_bindings_keep_fit tenv bindings = true ->
    carried (add_block_like_bindings tenv bindings).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros tenv Hcarried;
    cbn [block_like_bindings_keep_fit add_block_like_bindings];
    try (intros Hkeeps; exact (IH tenv Hcarried Hkeeps));
    [ intros _; exact Hcarried | ].
  destruct (is_closure_const const);
    [ intros Hkeeps; exact (IH tenv Hcarried Hkeeps) | ].
  pose proof (add_symbol_definition_carried tenv s Hcarried) as Hdefined.
  destruct (static_const_type_in tenv const) as [ty | ].
  - intros Hkeeps. apply andb_true_iff in Hkeeps.
    destruct Hkeeps as [Hequation Hrest]. apply IH; [ | exact Hrest ].
    exact (equation_keeps_fit_carried _ (Name_symbol s) ty Hdefined
             Hequation).
  - intros Hkeeps. exact (IH _ Hdefined Hkeeps).
Qed.

Lemma closure_bindings_keep_fit_carried (prior : Typing_env.t)
    (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    carried tenv ->
    closure_bindings_keep_fit prior tenv bindings = true ->
    carried (add_closure_bindings prior tenv bindings).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros tenv Hcarried;
    cbn [closure_bindings_keep_fit add_closure_bindings];
    try (intros Hkeeps; exact (IH tenv Hcarried Hkeeps));
    [ intros _; exact Hcarried | ].
  destruct (is_closure_const const);
    [ | intros Hkeeps; exact (IH tenv Hcarried Hkeeps) ].
  destruct (static_const_type_in prior const) as [ty | ];
    [ | intros Hkeeps; exact (IH tenv Hcarried Hkeeps) ].
  intros Hkeeps. apply andb_true_iff in Hkeeps.
  destruct Hkeeps as [Hequation Hrest]. apply IH; [ | exact Hrest ].
  exact (equation_keeps_fit_carried tenv (Name_symbol s) ty Hcarried
           Hequation).
Qed.

Theorem statics_keep_fit_sound : statics_keep_fit_sound_statement.
Proof.
  intros tenv fit bindings Hknown Hfit.
  apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
  unfold statics_keep_fit in Hkeeps. cbv zeta in Hkeeps.
  apply andb_true_iff in Hkeeps. destruct Hkeeps as [Hblocks Hclosures].
  unfold add_static_bindings. cbv zeta.
  apply closure_bindings_keep_fit_carried; [ | exact Hclosures ].
  apply block_like_bindings_keep_fit_carried; [ | exact Hblocks ].
  exact (add_symbol_definitions_carried tenv _ (Hknown Hfit)).
Qed.

Lemma define_variables_checked_carried
    (definitions : list (variable * flambda_kind)) :
  ∀ tenv tenv' : Typing_env.t,
    carried tenv ->
    define_variables_checked tenv definitions = Some tenv' ->
    carried tenv'.
Proof.
  induction definitions as [ | [x kind] definitions IH];
    intros tenv tenv' Hcarried; cbn [define_variables_checked].
  - intros [= <-]. exact Hcarried.
  - destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [_ Hfresh].
    apply negb_true_iff in Hfresh. apply IH.
    exact (add_variable_definition_carried tenv x kind Hcarried Hfresh).
Qed.

Theorem code_env_in_known_to_fit : code_env_in_known_to_fit_statement.
Proof.
  intros tenv fit params my_closure metadata tenv_code Hknown Henv Hfit.
  unfold code_env_in in Henv.
  destruct (Nat.eqb (length (unarize (cm_params_arity metadata)))
              (length params));
    [ | discriminate ].
  apply (define_variables_checked_carried _ _ tenv_code) in Henv;
    [ exact Henv | ].
  destruct (Hknown Hfit) as [Hinvariant Haliases]. split.
  - exact (closure_env_invariant tenv Hinvariant).
  - exact (closure_env_aliases_kinded tenv Haliases).
Qed.

Theorem handler_env_known_to_fit : handler_env_known_to_fit_statement.
Proof.
  intros tenv fit params metadata tenv_handler Hknown Henv Hfit.
  unfold handler_env in Henv.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | discriminate ].
  exact (define_variables_checked_carried _ tenv tenv_handler (Hknown Hfit)
           Henv).
Qed.

Theorem let_keeps_fit_sound : let_keeps_fit_sound_statement.
Proof.
  intros tenv tenv' fit x d d' Hknown Hlet Hfit.
  apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
  pose proof (Hknown Hfit) as Hcarried. revert Hlet Hkeeps.
  unfold simplify_let, let_keeps_fit.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
    [ | discriminate ].
  apply andb_true_iff in Htests. destruct Htests as [_ Hfresh].
  apply negb_true_iff in Hfresh.
  destruct (simplify_named tenv d) as [ | s | ty | kind | ]; try discriminate.
  - unfold bind_simple.
    destruct (simplify_simple tenv s) as [[ty canonical] | ];
      [ | discriminate ].
    destruct (simple_in_terms_b tenv canonical
              && simple_of_kind_b tenv canonical (FlambdaTypeGrammar.kind ty));
      [ | discriminate ].
    pose proof (add_variable_definition_carried tenv x
                  (FlambdaTypeGrammar.kind ty) Hcarried Hfresh) as Hcarried1.
    set (tenv1 :=
           Typing_env.add_variable_definition tenv x
             (FlambdaTypeGrammar.kind ty) Name_mode.normal) in *.
    destruct (Typing_env.add_alias tenv1 (Var x) canonical)
      as [[ | result | ] | ] eqn:Hadd; try discriminate.
    intros [= _ <-] Hkeeps. unfold alias_keeps_fit in Hkeeps.
    rewrite !andb_true_iff in Hkeeps.
    destruct Hkeeps as [[[[[Hc1 Hc2] Hm1] Hm2] Hk1] Hk2].
    destruct Hcarried1 as [Hinvariant1 Haliases1].
    apply is_canonical_b_iff in Hc1. apply is_canonical_b_iff in Hc2.
    pose proof (proj1 (typing_env_well_formed_iff _) (proj1 Hinvariant1))
      as Hconditions.
    destruct Hconditions as [_ [_ [Hconsistent _]]].
    split.
    + exact (add_alias_invariant tenv1 (Var x) canonical result Hinvariant1
               Hc1 Hc2 Hm1 Hm2 Hadd).
    + exact (add_alias_aliases_kinded tenv1 (Var x) canonical result
               (FlambdaTypeGrammar.kind ty) Hconsistent Hc1 Hc2
               (simple_of_kind_b_sound tenv1 _ _ Hk1)
               (simple_of_kind_b_sound tenv1 _ _ Hk2) Hadd Haliases1).
  - destruct (type_in_scope_b tenv ty); [ | discriminate ].
    pose proof (add_variable_definition_carried tenv x
                  (FlambdaTypeGrammar.kind ty) Hcarried Hfresh) as Hcarried1.
    set (tenv1 :=
           Typing_env.add_variable_definition tenv x
             (FlambdaTypeGrammar.kind ty) Name_mode.normal) in *.
    intros [= _ <-] Hkeeps.
    exact (equation_keeps_fit_carried tenv1 (Name_var x) ty Hcarried1
             Hkeeps).
  - intros [= _ <-] _.
    exact (add_variable_definition_carried tenv x kind Hcarried Hfresh).
Qed.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

(* After the step of a [Let] the second environment is satisfied: what
   the statements of FlambdaTypingEnvRefinementSpec.v about a [Let]
   show on their way, as they show it. *)
Definition let_establishes (tenv : Typing_env.t) (x : variable) (d : named)
    (tenv' : Typing_env.t) : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    env_satisfies_strictly tenv' statics ((x, v) :: vars).

Lemma let_establishes_kind (tenv : Typing_env.t) (x : variable) (d : named)
    (kind : flambda_kind) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
     (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    kind_of_value v = kind) ->
  let_establishes tenv x d
    (Typing_env.add_variable_definition tenv x kind Name_mode.normal).
Proof.
  intros Hwf Hfresh Hkind statics hp hp' rd rd' vars v Hsat Hv.
  apply add_variable_definition_env_satisfies_strictly;
    [ exact Hwf | exact Hfresh
    | exact (Hkind statics hp hp' rd rd' vars v Hsat Hv) | exact Hsat ].
Qed.

Lemma let_establishes_typed (tenv : Typing_env.t) (x : variable) (d : named)
    (ty : flambda_type) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
  (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
     (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    type_satisfied statics vars v ty) ->
  let_establishes tenv x d
    (Typing_env.replace_equation
       (Typing_env.add_variable_definition tenv x
          (FlambdaTypeGrammar.kind ty) Name_mode.normal)
       (Name_var x) ty).
Proof.
  intros Hwf Hfresh Hfree Htype statics hp hp' rd rd' vars v Hsat Hv.
  pose proof (Htype statics hp hp' rd rd' vars v Hsat Hv) as Hty.
  apply replace_equation_env_satisfies_strictly.
  - apply (add_variable_definition_in_scope tenv x _ x Hwf). left.
    reflexivity.
  - intros y Hin. apply (add_variable_definition_in_scope tenv x _ y Hwf).
    right. split; [ | exact (Hfree y Hin) ].
    exact (not_bound_not_in_scope tenv x y Hfresh (Hfree y Hin)).
  - intros v' Hlookup.
    pose proof (eq_trans (eq_sym Hlookup) (lookup_head x v vars)) as Heq.
    injection Heq as ->.
    apply (type_satisfied_coincide statics vars ((x, v) :: vars) v ty);
      [ | exact Hty ].
    intros y Hin. symmetry. apply lookup_other.
    exact (not_bound_not_in_scope tenv x y Hfresh (Hfree y Hin)).
  - apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | | exact Hsat ].
    exact (type_satisfied_kind statics vars v ty Hty).
Qed.

Lemma let_establishes_simple (tenv : Typing_env.t) (x : variable) (s : simple)
    (kind : flambda_kind) (result : Typing_env.add_alias_result) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  (∀ y : variable, s = Var y -> in_scope_in_terms tenv y) ->
  simple_of_kind tenv s kind ->
  Typing_env.add_alias
    (Typing_env.add_variable_definition tenv x kind Name_mode.normal) (Var x)
    s
  = Some (Ok result) ->
  let_establishes tenv x (Simple s) (Typing_env.add_alias_result_t result).
Proof.
  intros Hwf Hfresh Hscope Hkind Hadd statics hp hp' rd rd' vars v Hsat
    Hexec.
  assert (Hv : eval_simple vars s = Some v).
  { simpl in Hexec.
    destruct (eval_simple vars s) as [w | ]; [ | discriminate ].
    injection Hexec as -> _ _. reflexivity. }
  assert (Hother :
    eval_simple ((x, v) :: vars) s = eval_simple vars s).
  { destruct s as [y | sym | c]; [ | reflexivity.. ]. simpl.
    apply lookup_other.
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)). }
  apply (add_alias_env_satisfies_strictly
           (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
           statics ((x, v) :: vars) (Var x) s result).
  - intros y [[= <-] | ->];
      apply (add_variable_definition_in_scope tenv x kind _ Hwf);
      [ left; reflexivity | right ].
    split; [ | exact (Hscope y eq_refl) ].
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)).
  - exact Hadd.
  - rewrite Hother, Hv. simpl. apply lookup_head.
  - apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | | exact Hsat ].
    apply (simple_of_kind_value tenv statics vars s kind v);
      [ | exact Hscope | exact Hkind | exact Hv ].
    apply env_satisfies_strictly_env_satisfies. exact Hsat.
Qed.

(* The step of a [Let] keeps what [run_valid] says apart from the
   expression, on the heap and with the record that the step leaves. *)
Lemma rest_valid_let (x : variable) (declared : flambda_kind) (d : named)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (v : value) :
  named_sets_well_formed d = true ->
  sets_well_formed body = true ->
  rest_valid vars conts statics hp rd ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  kind_of_value v = declared ->
  rest_valid ((x, v) :: vars) conts statics hp' rd'.
Proof.
  intros Hd Hbody Hvalid Hv Hkind. apply (run_valid_rest body).
  refine
    (run_valid_kept _ _
       (rest_valid_run (Let x declared d body) vars conts statics hp rd _
          Hvalid)
       _).
  - cbn [sets_well_formed]. rewrite Hd, Hbody. reflexivity.
  - rewrite step_let_exec, Hv, (proj2 (kind_eqb_eq _ _) Hkind). reflexivity.
Qed.

(* The body of a [Let] runs in a state that has [run_valid] and
   satisfies the second environment. *)
Lemma let_body_refines_valid (tenv tenv' : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) (body body' : expr) :
  named_sets_well_formed d = true ->
  sets_well_formed body = true ->
  let_establishes tenv x d tenv' ->
  refines_valid tenv' body body' ->
  refines_valid tenv (Let x declared d body) (Let x declared d body').
Proof.
  intros Hd Hbody Hestablishes Hrefines vars conts statics hp rd Hvalid Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
  apply Hrefines;
    [ | exact (Hestablishes statics hp hp' rd rd' vars v Hsat Hv) ].
  exact (rest_valid_let x declared d body vars conts statics hp hp' rd rd' v
           Hd Hbody Hvalid Hv Hkind).
Qed.

(* The same at the kind that the [Let] declares, which the test of
   the machine gives. *)
Lemma let_declared_refines_valid (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) (body body' : expr) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  named_sets_well_formed d = true ->
  sets_well_formed body = true ->
  refines_valid
    (Typing_env.add_variable_definition tenv x declared Name_mode.normal)
    body body' ->
  refines_valid tenv (Let x declared d body) (Let x declared d body').
Proof.
  intros Hwf Hfresh Hd Hbody Hrefines vars conts statics hp rd Hvalid Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
  apply Hrefines.
  - exact (rest_valid_let x declared d body vars conts statics hp hp' rd rd'
             v Hd Hbody Hvalid Hv Hkind).
  - apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | exact Hkind | exact Hsat ].
Qed.

Lemma canonical_named_sets (tenv : Typing_env.t) (d : named) :
  named_sets_well_formed d = true ->
  named_sets_well_formed (canonical_named tenv d) = true.
Proof.
  destruct d; cbn [canonical_named]; try (intros H; exact H).
  intros _. destruct (simplify_simples tenv _); reflexivity.
Qed.

Lemma resolve_value_slot_sets (tenv : Typing_env.t) (declared : flambda_kind)
    (d : named) :
  named_sets_well_formed d = true ->
  named_sets_well_formed (resolve_value_slot tenv declared d) = true.
Proof.
  destruct d; cbn [resolve_value_slot]; try (intros H; exact H);
    intros _;
    repeat match goal with
           | |- context [match ?e with _ => _ end] => destruct e
           end;
    reflexivity.
Qed.

(* A [Let] that the function goes under: its new defining expression
   has well-formed sets, and the [Let] refines the new [Let] where the
   body refines the new body under the new environment. *)
Lemma simplify_let_valid (tenv tenv' : Typing_env.t) (x : variable)
    (d d' : named) :
  simplify_let tenv x d = Step_let d' tenv' ->
  named_sets_well_formed d = true ->
  named_sets_well_formed d' = true
  ∧ ∀ (declared : flambda_kind) (body body' : expr),
      sets_well_formed body = true ->
      refines_valid tenv' body body' ->
      refines_valid tenv (Let x declared d body) (Let x declared d' body').
Proof.
  intros Hlet Hd. revert Hlet. unfold simplify_let.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
    [ | discriminate ].
  apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
  apply negb_true_iff in Hfresh.
  pose proof (simplify_named_sound tenv d) as Hnamed.
  destruct (simplify_named tenv d) as [ | s | ty | kind | ]; simpl in Hnamed;
    try discriminate.
  - unfold bind_simple.
    destruct (simplify_simple tenv s) as [[ty canonical] | ] eqn:Hs;
      [ | discriminate ].
    destruct (simplify_simple_answer tenv s canonical ty Hs)
      as [_ Hcanonical].
    destruct (simple_in_terms_b tenv canonical
              && simple_of_kind_b tenv canonical (FlambdaTypeGrammar.kind ty))
      eqn:Htests; [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hterms Hkind].
    destruct (Typing_env.add_alias _ _ _) as [[ | result | ] | ] eqn:Hadd;
      try discriminate.
    intros [= <- <-]. split; [ reflexivity | ].
    intros declared body body' Hbody Hrefines.
    apply (refines_valid_trans tenv _
             (Let x declared (Simple canonical) body)).
    + apply refines_under_valid.
      eapply expr_refines_under_trans; [ exact (Hnamed x declared body) | ].
      apply expr_equivalent_under_refines.
      exact (let_simple_canonical tenv x declared s canonical body
               Hcanonical).
    + eapply (let_body_refines_valid tenv _ x declared (Simple canonical) body
               body' eq_refl Hbody);
        [ | exact Hrefines ].
      exact (let_establishes_simple tenv x canonical
               (FlambdaTypeGrammar.kind ty) result Hwf Hfresh
               (simple_in_terms_b_sound tenv canonical Hterms)
               (simple_of_kind_b_sound tenv canonical _ Hkind) Hadd).
  - destruct (type_in_scope_b tenv ty) eqn:Hscope; [ | discriminate ].
    intros [= <- <-]. split; [ exact (canonical_named_sets tenv d Hd) | ].
    intros declared body body' Hbody Hrefines.
    apply (refines_valid_trans tenv _ (Let x declared d body')).
    + eapply (let_body_refines_valid tenv _ x declared d body body' Hd Hbody);
        [ | exact Hrefines ].
      apply (let_establishes_typed tenv x d ty Hwf Hfresh);
        [ | exact Hnamed ].
      intros y Hy. unfold type_in_scope_b in Hscope.
      rewrite forallb_forall in Hscope. exact (Hscope y Hy).
    + apply refines_under_valid, expr_equivalent_under_refines.
      exact (canonical_named_equivalent tenv x declared d body').
  - intros [= <- <-]. split; [ exact (canonical_named_sets tenv d Hd) | ].
    intros declared body body' Hbody Hrefines.
    apply (refines_valid_trans tenv _ (Let x declared d body')).
    + eapply (let_body_refines_valid tenv _ x declared d body body' Hd Hbody);
        [ | exact Hrefines ].
      exact (let_establishes_kind tenv x d kind Hwf Hfresh Hnamed).
    + apply refines_under_valid, expr_equivalent_under_refines.
      exact (canonical_named_equivalent tenv x declared d body').
Qed.

(* ======================================================================== *)
(* A [Switch]                                                               *)
(* ======================================================================== *)

Theorem simplify_switch_rewrite_sound
    : simplify_switch_rewrite_sound_statement.
Proof.
  intros meet Hmeet tenv s arms e He. split.
  - revert He. unfold FlambdaSimplifySwitchSpec.simplify_switch.
    destruct (Typing_env.type_simple_in_term_exn tenv
                (Some Name_mode.normal) s)
      as [[[ty canonical] | ] | ]; try discriminate.
    cbn.
    destruct (FlambdaSimplifySwitchSpec.prune_arms meet tenv ty arms)
      as [arms' | ]; [ | discriminate ].
    intros [= <-]. destruct arms' as [ | [i ac] [ | arm arms']]; reflexivity.
  - intros Hfits Hmem.
    exact (FlambdaSimplifySwitchProofs.simplify_switch_refines meet tenv s
             arms e Hmeet Hfits Hmem He).
Qed.

Lemma simplify_switch_sets (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  sets_well_formed (simplify_switch tenv scrutinee arms) = true.
Proof.
  unfold simplify_switch.
  repeat match goal with
         | |- context [match ?e with _ => _ end] => destruct e
         end;
    reflexivity.
Qed.

Lemma after_pruning_sound (tenv : Typing_env.t) (e : expr) :
  sets_well_formed e = true ->
  sets_well_formed (after_pruning tenv e) = true
  ∧ expr_refines_under tenv e (after_pruning tenv e).
Proof.
  intros Hsets.
  destruct e
    as [x declared d body | bindings body | k handler body
        | handlers body | ap | ac | scrutinee arms | ];
    cbn [after_pruning];
    try (split; [ exact Hsets | apply expr_refines_under_refl ]).
  - split; [ reflexivity | ].
    apply expr_equivalent_under_refines, simplify_apply_cont_equivalent.
  - split; [ apply simplify_switch_sets | apply simplify_switch_refines ].
Qed.

Lemma simplify_switch_pruning_sound (prune : switch_rewrite)
    (tenv : Typing_env.t) (fit : bool) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  switch_rewrite_sound prune ->
  known_to_fit tenv fit ->
  sets_well_formed (simplify_switch_pruning prune tenv fit scrutinee arms)
  = true
  ∧ refines_valid tenv (Switch scrutinee arms)
      (simplify_switch_pruning prune tenv fit scrutinee arms).
Proof.
  intros Hprune Hknown.
  assert (Hfirst :
    sets_well_formed (simplify_switch tenv scrutinee arms) = true
    ∧ refines_valid tenv (Switch scrutinee arms)
        (simplify_switch tenv scrutinee arms)).
  { split; [ apply simplify_switch_sets | ].
    apply refines_under_valid, simplify_switch_refines. }
  unfold simplify_switch_pruning.
  destruct (fit && types_fit_b tenv
            && Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee)
    eqn:Htests; [ | exact Hfirst ].
  destruct (prune tenv scrutinee arms) as [e | ] eqn:He; [ | exact Hfirst ].
  apply andb_true_iff in Htests. destruct Htests as [Htests Hmem].
  apply andb_true_iff in Htests. destruct Htests as [Hfit Htypes].
  destruct (Hknown Hfit) as [Hinvariant Haliases].
  destruct (Hprune tenv scrutinee arms e He) as [Hsets Hrefines].
  destruct (after_pruning_sound tenv e Hsets) as [Hsets' Hafter].
  split; [ exact Hsets' | ].
  apply (refines_valid_trans tenv _ e).
  - apply refines_valid_of_under_valid. apply Hrefines; [ | exact Hmem ].
    exact (types_fit_b_fits tenv Hinvariant Haliases Htypes).
  - apply refines_under_valid. exact Hafter.
Qed.

(* ======================================================================== *)
(* The other binders                                                        *)
(* ======================================================================== *)

Lemma let_cont_body_refines_valid (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body body' : expr) :
  sets_well_formed (Let_cont_non_recursive k h body) = true ->
  refines_valid tenv body body' ->
  refines_valid tenv (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h body').
Proof.
  intros Hsets Hbody vars conts statics hp rd Hvalid Hsat.
  apply let_cont_non_recursive_refines_step. apply Hbody; [ | exact Hsat ].
  exact (rest_valid_step _ body vars _ conts _ statics _ _ rd_ Hsets Hvalid
           (step_let_cont_non_recursive k h body vars conts statics hp rd)).
Qed.

Lemma let_cont_recursive_body_refines_valid (tenv : Typing_env.t)
    (handlers : list (continuation * continuation_handler))
    (body body' : expr) :
  sets_well_formed (Let_cont_recursive handlers body) = true ->
  refines_valid tenv body body' ->
  refines_valid tenv (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers body').
Proof.
  intros Hsets Hbody vars conts statics hp rd Hvalid Hsat.
  apply let_cont_recursive_refines_step; [ reflexivity | intros Hbinds ].
  apply Hbody; [ | exact Hsat ].
  apply (rest_valid_step _ body vars _ conts _ statics _ _ rd_ Hsets Hvalid).
  rewrite step_let_cont_recursive, Hbinds. reflexivity.
Qed.

(* A handler rewritten under its environment
   (FlambdaSimplifyHandlersSpec.let_cont_handler_typed), for refinement
   from the states that have [run_valid]. The predicate is [run_valid]
   with that the static environment extends the one at the
   [Let_cont]. *)
Lemma handler_typed_from (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr) (tenv_handler : Typing_env.t) :
  handler_env tenv params metadata = Some tenv_handler ->
  sets_well_formed handler = true ->
  sets_well_formed handler' = true ->
  refines_valid tenv_handler handler handler' ->
  env_satisfies_strictly tenv statics vars ->
  handler_refines_from
    (fun s => run_valid s ∧ statics_extend statics (state_statics s)) params
    (hm_params_kinds metadata) vars handler handler'.
Proof.
  intros Henv Hsets Hsets' Hhandler Hsat vs handler_vars conts' statics' hp rd
    Hkinds Hbind.
  split.
  - split; intros [Hrun Hextend]; (split; [ | exact Hextend ]).
    + exact (run_valid_expr _ _ _ _ _ _ rd_ Hsets' Hrun).
    + exact (run_valid_expr _ _ _ _ _ _ rd_ Hsets Hrun).
  - intros [Hrun Hextend]. apply Hhandler.
    + exact (run_valid_rest _ _ _ _ _ rd_ Hrun).
    + exact (handler_env_satisfied tenv params metadata tenv_handler
               statics' vars vs handler_vars Henv
               (env_satisfies_strictly_statics_extend tenv statics statics'
                  vars Hextend Hsat)
               Hkinds Hbind).
Qed.

Lemma valid_extending_kept (statics : static_env) :
  kept_by_step
    (fun s => run_valid s ∧ statics_extend statics (state_statics s)).
Proof.
  intros s s' [Hrun Hextend] Hstep.
  split; [ exact (run_valid_kept s s' Hrun Hstep) | ].
  exact (statics_extend_trans statics _ _ Hextend
           (step_extends_statics s s' Hstep)).
Qed.

Lemma let_cont_handler_typed_valid (tenv : Typing_env.t) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (tenv_handler : Typing_env.t) :
  handler_env tenv params metadata = Some tenv_handler ->
  sets_well_formed handler = true ->
  sets_well_formed handler' = true ->
  sets_well_formed body = true ->
  refines_valid tenv_handler handler handler' ->
  refines_valid tenv
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Henv Hsets Hsets' Hbody Hhandler vars conts statics hp rd Hvalid Hsat.
  apply (FlambdaContextWhereProofs.let_cont_handler_refines_from
           (fun s => run_valid s ∧ statics_extend statics (state_statics s))).
  - exact (valid_extending_kept statics).
  - split; [ | apply statics_extend_refl ].
    apply rest_valid_run; [ | exact Hvalid ].
    cbn [sets_well_formed handler_sets_well_formed].
    rewrite Hsets, Hbody. reflexivity.
  - split; [ | apply statics_extend_refl ].
    apply rest_valid_run; [ | exact Hvalid ].
    cbn [sets_well_formed handler_sets_well_formed].
    rewrite Hsets', Hbody. reflexivity.
  - exact (handler_typed_from tenv statics vars params metadata handler
             handler' tenv_handler Henv Hsets Hsets' Hhandler Hsat).
Qed.

Lemma let_cont_recursive_handler_typed_valid (tenv : Typing_env.t)
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr)
    (tenv_handler : Typing_env.t) :
  handler_env tenv params metadata = Some tenv_handler ->
  binds_statics handler = binds_statics handler' ->
  sets_well_formed handler = true ->
  sets_well_formed handler' = true ->
  sets_well_formed
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
  = true ->
  sets_well_formed
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler') :: after) body)
  = true ->
  refines_valid tenv_handler handler handler' ->
  refines_valid tenv
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler') :: after) body).
Proof.
  intros Henv Hbinds Hsets Hsets' Hall Hall' Hhandler vars conts statics hp rd
    Hvalid Hsat.
  apply (FlambdaContextWhereProofs.let_cont_recursive_handler_refines_from
           (fun s => run_valid s ∧ statics_extend statics (state_statics s))).
  - exact (valid_extending_kept statics).
  - split; [ | apply statics_extend_refl ].
    exact (rest_valid_run _ vars conts statics _ rd_ Hall Hvalid).
  - split; [ | apply statics_extend_refl ].
    exact (rest_valid_run _ vars conts statics _ rd_ Hall' Hvalid).
  - exact Hbinds.
  - exact (handler_typed_from tenv statics vars params metadata handler
             handler' tenv_handler Henv Hsets Hsets' Hhandler Hsat).
Qed.

(* The sets of a recursive [Let_cont], one handler apart. *)
Lemma let_cont_recursive_sets
    (before after : list (continuation * continuation_handler))
    (k : continuation) (h : continuation_handler) (body : expr) :
  sets_well_formed (Let_cont_recursive (before ++ (k, h) :: after) body)
  = handler_sets_well_formed h
    && sets_well_formed (Let_cont_recursive (before ++ after) body).
Proof.
  cbn [sets_well_formed]. rewrite !forallb_app. cbn [forallb].
  destruct (forallb (fun '(_, h0) => handler_sets_well_formed h0) before),
    (handler_sets_well_formed h),
    (forallb (fun '(_, h0) => handler_sets_well_formed h0) after),
    (sets_well_formed body);
    reflexivity.
Qed.

Lemma let_static_body_refines_valid (tenv : Typing_env.t)
    (bindings : list static_binding) (body body' : expr) :
  sets_well_formed (Let_static bindings body) = true ->
  refines_valid (add_static_bindings tenv bindings) body body' ->
  refines_valid tenv (Let_static bindings body) (Let_static bindings body').
Proof.
  intros Hsets Hbody vars conts statics hp rd Hvalid Hsat.
  destruct (eval_static_bindings vars bindings statics) as [statics' | ]
    eqn:Hbindings.
  - assert (Hstep :
      ∀ b : expr,
        step (State (Let_static bindings b) vars conts statics hp rd)
        = Step (State b vars conts statics' hp rd)).
    { intros b. rewrite step_let_static, Hbindings. reflexivity. }
    eapply refines_step; [ apply Hstep | apply Hstep | ].
    apply Hbody.
    + exact (rest_valid_step _ body vars _ conts _ statics _ _ rd_ Hsets Hvalid
               (Hstep body)).
    + exact (add_static_bindings_env_satisfies tenv statics statics' vars
               bindings Hbindings Hsat).
  - apply step_undefined_refines. rewrite step_let_static, Hbindings.
    reflexivity.
Qed.

Lemma add_static_bindings_simplify_pruning (prune : switch_rewrite)
    (tenv tenv' : Typing_env.t) (fit : bool)
    (bindings : list static_binding) :
  add_static_bindings tenv
    (map (simplify_static_binding_pruning prune tenv' fit) bindings)
  = add_static_bindings tenv bindings.
Proof.
  apply add_static_bindings_same_symbols.
  - intros s const. reflexivity.
  - intros id c. eexists. reflexivity.
  - intros id. reflexivity.
Qed.

(* The body of code that a [Let_static] binds, rewritten under the
   environment of the code made from the environment of the body of
   the [Let_static]: FlambdaSimplifyCodeEnvSpec.let_static_code_typed
   for refinement from the states that have [run_valid]. The predicate
   is [run_valid] with what the proof of that statement carries: the
   static environment satisfies the environment outside the code, or
   the state is about to execute a group that will make it so. *)
Lemma let_static_code_typed_valid (tenv : Typing_env.t)
    (before : list static_binding) (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (code_body code_body' : expr)
    (after : list static_binding) (body : expr) (tenv_code : Typing_env.t) :
  code_env_in
    (add_static_bindings tenv
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body)
        :: after))
    params my_closure metadata
  = Some tenv_code ->
  binds_statics code_body = false ->
  binds_statics code_body' = false ->
  sets_well_formed code_body = true ->
  sets_well_formed code_body' = true ->
  sets_well_formed
    (Let_static
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body)
        :: after)
       body)
  = true ->
  sets_well_formed
    (Let_static
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body')
        :: after)
       body)
  = true ->
  refines_valid tenv_code code_body code_body' ->
  refines_valid tenv
    (Let_static
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body)
        :: after)
       body)
    (Let_static
       (before
        ++ Bind_code id (Code params my_closure ret exn metadata code_body')
        :: after)
       body).
Proof.
  intros Henv Hsource Htarget Hcode Hcode' Hsets Hsets' Hbody vars conts
    statics hp' rd' Hvalid Hsat.
  pose (source :=
          before
          ++ Bind_code id (Code params my_closure ret exn metadata code_body)
          :: after).
  pose (target :=
          before
          ++ Bind_code id
               (Code params my_closure ret exn metadata code_body')
          :: after).
  pose (outer := Typing_env.closure_env (add_static_bindings tenv source)).
  pose (inv :=
          fun s : state =>
            env_satisfies_strictly outer (state_statics s) []
            ∨ ∃ (bindings : list static_binding) (b : expr),
                state_expr s = Let_static bindings b
                ∧ existsb binding_binds bindings = true
                ∧ ∀ extended : static_env,
                    eval_static_bindings (state_vars s) bindings
                      (state_statics s)
                    = Some extended ->
                    env_satisfies_strictly outer extended []).
  assert (Hwf : typing_env_well_formed outer = true).
  { exact (code_env_in_well_formed _ params my_closure metadata tenv_code
             Henv). }
  assert (Hgroup :
    ∀ bindings : list static_binding,
      add_static_bindings tenv bindings = add_static_bindings tenv source ->
      ∀ extended : static_env,
        eval_static_bindings vars bindings statics = Some extended ->
        env_satisfies_strictly outer extended []).
  { intros bindings Heq extended Heval.
    apply (closure_env_env_satisfies_strictly
             (add_static_bindings tenv source) extended vars Hwf).
    rewrite <- Heq.
    exact (add_static_bindings_env_satisfies tenv statics extended vars
             bindings Heval Hsat). }
  assert (Hcall :
    ∀ (e : expr) (vars0 : env) (conts0 : cont_env) (statics0 : static_env)
      (hp : heap) (rd : read_record),
      binds_statics e = false ->
      inv (State e vars0 conts0 statics0 hp rd) ->
      env_satisfies_strictly outer statics0 []).
  { intros e vars0 conts0 statics0 hp rd He Hinv. unfold inv in Hinv.
    destruct Hinv as [Hsat0 | [bindings [b [Hexpr [Hbinds _]]]]];
      [ exact Hsat0 | ].
    simpl in Hexpr. subst e. cbn [binds_statics] in He.
    rewrite Hbinds in He. discriminate. }
  apply (FlambdaContextWhereProofs.let_static_code_refines_from
           (fun s => run_valid s ∧ inv s) before id params my_closure ret
           exn metadata code_body code_body' after body vars conts statics).
  - intros s s' [Hrun Hinv] Hstep.
    split; [ exact (run_valid_kept s s' Hrun Hstep) | ].
    unfold inv in Hinv |- *.
    destruct Hinv as [Hsat0 | [bindings [b [Hexpr [_ Hextended]]]]].
    + left.
      exact (env_satisfies_strictly_statics_extend outer (state_statics s)
               (state_statics s') [] (step_extends_statics s s' Hstep) Hsat0).
    + left. destruct s as [e vars0 conts0 statics0 hp rd].
      simpl in Hexpr, Hextended. subst e.
      rewrite step_let_static in Hstep.
      destruct (eval_static_bindings vars0 bindings statics0)
        as [extended | ] eqn:Heval; [ | discriminate ].
      injection Hstep as <-. exact (Hextended extended eq_refl).
  - split; [ exact (rest_valid_run _ vars conts statics _ rd_ Hsets Hvalid) | ].
    unfold inv. right. exists source, body.
    split; [ reflexivity | ]. split.
    + unfold source. rewrite existsb_app. simpl. apply orb_true_r.
    + exact (Hgroup source eq_refl).
  - split;
    [ exact (rest_valid_run _ vars conts statics _ rd_ Hsets' Hvalid) | ].
    unfold inv. right. exists target, body.
    split; [ reflexivity | ]. split.
    + unfold target. rewrite existsb_app. simpl. apply orb_true_r.
    + apply Hgroup. apply add_static_bindings_code.
  - rewrite Hsource, Htarget. reflexivity.
  - intros vs callee callee_vars return_to exn_to statics0 hp rd Hbind Hkinds
      Hcallee.
    split.
    + split; intros [Hrun Hinv]; split.
      * exact (run_valid_expr _ _ _ _ _ _ rd_ Hcode' Hrun).
      * unfold inv. left. exact (Hcall _ _ _ _ _ rd_ Hsource Hinv).
      * exact (run_valid_expr _ _ _ _ _ _ rd_ Hcode Hrun).
      * unfold inv. left. exact (Hcall _ _ _ _ _ rd_ Htarget Hinv).
    + intros [Hrun Hinv]. apply Hbody.
      * exact (run_valid_rest _ _ _ _ _ rd_ Hrun).
      * exact (code_env_in_satisfied_from_closure _ params my_closure
                 metadata tenv_code statics0 vs callee callee_vars Henv
                 (Hcall _ _ _ _ _ rd_ Hsource Hinv) Hkinds Hcallee Hbind).
Qed.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

Lemma alias_keeps_fit_carried (tenv : Typing_env.t) (e1 e2 : simple)
    (kind : flambda_kind) (result : Typing_env.add_alias_result) :
  carried tenv ->
  alias_keeps_fit tenv e1 e2 kind = true ->
  Typing_env.add_alias tenv e1 e2 = Some (Ok result) ->
  carried (Typing_env.add_alias_result_t result).
Proof.
  intros [Hinvariant Haliases] Hkeeps Hadd.
  unfold alias_keeps_fit in Hkeeps. rewrite !andb_true_iff in Hkeeps.
  destruct Hkeeps as [[[[[Hc1 Hc2] Hm1] Hm2] Hk1] Hk2].
  apply is_canonical_b_iff in Hc1. apply is_canonical_b_iff in Hc2.
  pose proof (proj1 (typing_env_well_formed_iff _) (proj1 Hinvariant))
    as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  split.
  - exact (add_alias_invariant tenv e1 e2 result Hinvariant Hc1 Hc2 Hm1 Hm2
             Hadd).
  - exact (add_alias_aliases_kinded tenv e1 e2 result kind Hconsistent Hc1
             Hc2 (simple_of_kind_b_sound tenv _ _ Hk1)
             (simple_of_kind_b_sound tenv _ _ Hk2) Hadd Haliases).
Qed.

Lemma arm_env_carried (tenv : Typing_env.t) (scrutinee : simple) (d : Z) :
  carried tenv ->
  arm_env_keeps_fit tenv scrutinee d = true ->
  carried (arm_env tenv scrutinee d).
Proof.
  intros Hcarried. unfold arm_env_keeps_fit, arm_env.
  destruct (simplify_simple tenv scrutinee) as [[ty [c | sym | const]] | ];
    try (intros _; exact Hcarried).
  destruct (simple_in_terms_b tenv (Var c)
            && simple_of_kind_b tenv (Var c) naked_immediate_kind);
    [ | intros _; exact Hcarried ].
  destruct (Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)))
    as [[ | result | ] | ] eqn:Hadd; try (intros _; exact Hcarried).
  intros Hkeeps.
  exact (alias_keeps_fit_carried tenv _ _ _ result Hcarried Hkeeps Hadd).
Qed.

Lemma bind_params_to_args_carried (aliased : simple -> bool)
    (bindings : list (variable * flambda_kind * simple)) :
  ∀ tenv target : Typing_env.t,
    carried tenv ->
    bind_params_to_args aliased tenv bindings = Some target ->
    bind_params_to_args_keep_fit aliased tenv bindings = true ->
    carried target.
Proof.
  induction bindings as [ | [[x kind] canonical] bindings IH];
    intros tenv target Hcarried Hbind Hkeeps.
  - simpl in Hbind. injection Hbind as <-. exact Hcarried.
  - cbn [bind_params_to_args] in Hbind.
    cbn [bind_params_to_args_keep_fit] in Hkeeps.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_prop in Htests. destruct Htests as [_ Hfresh].
    apply negb_true_iff in Hfresh.
    pose proof (add_variable_definition_carried tenv x kind Hcarried Hfresh)
      as Hdefined.
    destruct (aliased canonical && simple_in_terms_b tenv canonical
              && simple_of_kind_b tenv canonical kind);
      [ | exact (IH _ target Hdefined Hbind Hkeeps) ].
    destruct (Typing_env.add_alias
                (Typing_env.add_variable_definition tenv x kind
                   Name_mode.normal)
                (Var x) canonical)
      as [[ | result | ] | ] eqn:Hadd;
      try exact (IH _ target Hdefined Hbind Hkeeps).
    apply andb_true_iff in Hkeeps. destruct Hkeeps as [Halias Hrest].
    exact (IH _ target
             (alias_keeps_fit_carried _ _ _ _ result Hdefined Halias Hadd)
             Hbind Hrest).
Qed.

Lemma use_env_carried (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple)
    (target : Typing_env.t) :
  carried tenv ->
  use_env scope tenv params metadata args = Some target ->
  use_env_keeps_fit scope tenv params metadata args = true ->
  carried target.
Proof.
  intros Hcarried. unfold use_env, use_env_keeps_fit.
  destruct (simplify_simples tenv args) as [answers | ]; [ | discriminate ].
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params)
            && Nat.eqb (length answers) (length params));
    [ | discriminate ].
  exact (bind_params_to_args_carried _ _ tenv target Hcarried).
Qed.

Lemma leaf_uses_carried (tenv : Typing_env.t) (k : continuation) (e : expr)
    (applications : list (Typing_env.t * list simple))
    (tenv_use : Typing_env.t) (args : list simple) :
  carried tenv ->
  leaf_uses tenv k e = Some applications ->
  leaf_uses_keep_fit tenv k e = true ->
  (tenv_use, args) ∈ applications ->
  carried tenv_use.
Proof.
  intros Hcarried.
  destruct e
    as [x declared d body | bindings body | k' h body | handlers body | ap
        | ac | scrutinee arms | ];
    simpl; try discriminate.
  - intros [= <-] _. destruct (Pos.eqb (ac_continuation ac) k);
      [ | intros [] ].
    intros [[= <- _] | []]. exact Hcarried.
  - intros [= <-] Hkeeps Hin. apply in_flat_map in Hin.
    destruct Hin as [[d ac] [Harm Hin]]. cbn [fst snd] in Hin.
    rewrite forallb_forall in Hkeeps. specialize (Hkeeps (d, ac) Harm).
    cbn [fst snd] in Hkeeps.
    destruct (Pos.eqb (ac_continuation ac) k); [ | destruct Hin ].
    destruct Hin as [[= <- _] | []]. cbn [negb orb] in Hkeeps.
    exact (arm_env_carried tenv scrutinee d Hcarried Hkeeps).
Qed.

Lemma let_env_known_to_fit (tenv tenv' : Typing_env.t) (fit : bool)
    (x : variable) (declared : flambda_kind) (d : named) :
  known_to_fit tenv fit ->
  let_env tenv x declared d = Some tenv' ->
  known_to_fit tenv' (fit && let_env_keeps_fit tenv x declared d).
Proof.
  intros Hknown. unfold let_env, let_env_keeps_fit. cbv zeta.
  destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
    as [ | d' tenv'' | ] eqn:Hlet; [ discriminate | | ].
  - intros [= <-].
    exact (let_keeps_fit_sound tenv tenv'' fit x _ d' Hknown Hlet).
  - unfold declare_variable.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    intros [= <-] Hfit. apply andb_true_iff in Hfit.
    destruct Hfit as [Hfit _]. apply andb_prop in Htests.
    destruct Htests as [_ Hfresh]. apply negb_true_iff in Hfresh.
    exact (add_variable_definition_carried tenv x declared (Hknown Hfit)
             Hfresh).
Qed.

Fixpoint find_use_known_to_fit_aux (body : expr) {struct body} :
  ∀ (scope tenv : Typing_env.t) (fit : bool) (k : continuation)
    (h : continuation_handler) (target : Typing_env.t) (xs : list variable)
    (ks : list continuation),
    known_to_fit tenv fit ->
    find_use scope tenv k h body = Some (target, xs, ks) ->
    known_to_fit target (fit && find_use_keeps_fit scope tenv k h body).
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros scope tenv fit k h target xs ks Hknown Hfind;
    cbn [find_use] in Hfind; cbn [find_use_keeps_fit]; try discriminate.
  - destruct (var_free_in_handler_b x h); [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (find_use scope tenv' k h rest) as [[[target' xs'] ks'] | ]
      eqn:Hrest; [ | discriminate ].
    injection Hfind as <- _ _.
    pose proof (find_use_known_to_fit_aux rest scope tenv'
                  (fit && let_env_keeps_fit tenv x declared d) k h target'
                  xs' ks'
                  (let_env_known_to_fit tenv tenv' fit x declared d Hknown
                     Henv)
                  Hrest)
      as IH.
    rewrite andb_assoc. exact IH.
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k h')
              && negb (cont_free_in_handler_b k' h));
      [ | discriminate ].
    destruct (find_use scope tenv k h rest) as [[[target' xs'] ks'] | ]
      eqn:Hrest; [ | discriminate ].
    injection Hfind as <- _ _.
    exact (find_use_known_to_fit_aux rest scope tenv fit k h target' xs' ks'
             Hknown Hrest).
  - destruct h as [params metadata handler].
    destruct (leaf_uses tenv k (Apply_cont ac))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try discriminate.
    destruct (use_env scope tenv_use params metadata args) as [target' | ]
      eqn:Huse; [ | discriminate ].
    injection Hfind as <- _ _. intros Hfit.
    apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
    apply andb_true_iff in Hkeeps. destruct Hkeeps as [Hleaf_fit Huse_fit].
    apply (use_env_carried scope tenv_use params metadata args target');
      [ | exact Huse | exact Huse_fit ].
    apply (leaf_uses_carried tenv k (Apply_cont ac) _ tenv_use args
             (Hknown Hfit) Hleaf Hleaf_fit).
    left. reflexivity.
  - destruct h as [params metadata handler].
    destruct (leaf_uses tenv k (Switch scrutinee arms))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try discriminate.
    destruct (use_env scope tenv_use params metadata args) as [target' | ]
      eqn:Huse; [ | discriminate ].
    injection Hfind as <- _ _. intros Hfit.
    apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
    apply andb_true_iff in Hkeeps. destruct Hkeeps as [Hleaf_fit Huse_fit].
    apply (use_env_carried scope tenv_use params metadata args target');
      [ | exact Huse | exact Huse_fit ].
    apply (leaf_uses_carried tenv k (Switch scrutinee arms) _ tenv_use args
             (Hknown Hfit) Hleaf Hleaf_fit).
    left. reflexivity.
Qed.

Theorem find_use_known_to_fit : find_use_known_to_fit_statement.
Proof.
  intros scope tenv fit k h body.
  exact (find_use_known_to_fit_aux body scope tenv fit k h).
Qed.

(* [FlambdaSimplifyPassProofs.find_use_refines_aux], from the states
   that have [run_valid] but for the test of the expression. *)
Fixpoint find_use_refines_valid (body : expr) {struct body} :
  ∀ (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' : expr)
    (target : Typing_env.t) (xs : list variable) (ks : list continuation),
    find_use scope tenv k (Handler params metadata handler) body
    = Some (target, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    sets_well_formed handler = true ->
    sets_well_formed body = true ->
    refines_valid target handler handler' ->
    refines_valid tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros scope tenv k params metadata handler handler' target xs ks Hfind
      Hfree Hhandler_sets Hbody_sets Hhandler;
    cbn [find_use] in Hfind; try discriminate.
  - destruct (var_free_in_handler_b x (Handler params metadata handler))
      eqn:Hx; [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (find_use scope tenv' k (Handler params metadata handler) rest)
      as [[[target' xs'] ks'] | ] eqn:Hrest; [ | discriminate ].
    injection Hfind as <- <- <-.
    unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
    apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
    apply andb_true_iff in Hxs. destruct Hxs as [Hx' Hxs].
    apply negb_true_iff in Hx'.
    assert (Hrest_free :
      binders_not_free xs' ks' (Handler params metadata handler') = true).
    { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
    cbn [sets_well_formed] in Hbody_sets.
    apply andb_true_iff in Hbody_sets. destruct Hbody_sets as [Hd Hrest_sets].
    pose proof (find_use_refines_valid rest scope tenv' k params metadata
                  handler handler' target' xs' ks' Hrest Hrest_free
                  Hhandler_sets Hrest_sets Hhandler)
      as IH.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest))
                vars conts statics hp rd)).
    { apply equivalent_refines, let_cont_past_let. intros Hfree'.
      apply var_free_in_handler_b_sound in Hfree'. congruence. }
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler') rest))
                vars conts statics hp rd)).
    { apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
      apply IH.
      - refine
          (rest_valid_let x declared d
             (Let_cont_non_recursive k (Handler params metadata handler)
                rest)
             vars conts statics hp hp' rd rd' v Hd _ Hvalid Hv Hkind).
        cbn [sets_well_formed handler_sets_well_formed].
        rewrite Hhandler_sets, Hrest_sets. reflexivity.
      - exact (let_env_satisfied tenv tenv' x declared d statics hp hp' rd
                 rd' vars v Henv Hsat Hv Hkind). }
    apply equivalent_refines, equivalent_sym, let_cont_past_let.
    intros Hfree'. apply var_free_in_handler_b_sound in Hfree'. congruence.
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k h')
              && negb
                   (cont_free_in_handler_b k'
                      (Handler params metadata handler)))
      eqn:Htests; [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Htests Hk'].
    apply andb_true_iff in Htests. destruct Htests as [Hne Hk].
    apply negb_true_iff in Hne, Hk, Hk'. apply Pos.eqb_neq in Hne.
    destruct (find_use scope tenv k (Handler params metadata handler) rest)
      as [[[target' xs'] ks'] | ] eqn:Hrest; [ | discriminate ].
    injection Hfind as <- <- <-.
    unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
    apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
    apply andb_true_iff in Hks. destruct Hks as [Hk'' Hks].
    apply negb_true_iff in Hk''.
    assert (Hrest_free :
      binders_not_free xs' ks' (Handler params metadata handler') = true).
    { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
    cbn [sets_well_formed] in Hbody_sets.
    apply andb_true_iff in Hbody_sets.
    destruct Hbody_sets as [Hh' Hrest_sets].
    pose proof (find_use_refines_valid rest scope tenv k params metadata
                  handler handler' target' xs' ks' Hrest Hrest_free
                  Hhandler_sets Hrest_sets Hhandler)
      as IH.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (refines_trans _
             (State
                (Let_cont_non_recursive k' h'
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest))
                vars conts statics hp rd)).
    { apply equivalent_refines, let_cont_past_let_cont; [ exact Hne | | ];
        intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
        congruence. }
    apply (refines_trans _
             (State
                (Let_cont_non_recursive k' h'
                   (Let_cont_non_recursive k
                      (Handler params metadata handler') rest))
                vars conts statics hp rd)).
    { apply let_cont_non_recursive_refines_step. apply IH; [ | exact Hsat ].
      eapply (rest_valid_step
                (Let_cont_non_recursive k' h'
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest)));
        [ | exact Hvalid | apply step_let_cont_non_recursive ].
      cbn [sets_well_formed handler_sets_well_formed].
      rewrite Hh', Hhandler_sets, Hrest_sets. reflexivity. }
    apply equivalent_refines, equivalent_sym, let_cont_past_let_cont;
      [ exact Hne | | ];
      intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
      congruence.
  - destruct (leaf_uses tenv k (Apply_cont ac))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try discriminate.
    destruct (use_env scope tenv_use params metadata args) as [target' | ]
      eqn:Huse; [ | discriminate ].
    injection Hfind as <- _ _.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (let_cont_handler_at_use_from run_valid scope tenv k params
             metadata handler handler' _ tenv_use args target'
             run_valid_kept Hleaf Huse);
      [ | | exact Hsat ].
    + intros vars' conts' statics' hp' rd' Hrun Hsat'.
      apply Hhandler; [ | exact Hsat' ].
      exact (run_valid_rest handler vars' conts' statics' _ _ Hrun).
    + apply rest_valid_run; [ | exact Hvalid ].
      cbn [sets_well_formed handler_sets_well_formed].
      rewrite Hhandler_sets. reflexivity.
  - destruct (leaf_uses tenv k (Switch scrutinee arms))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try discriminate.
    destruct (use_env scope tenv_use params metadata args) as [target' | ]
      eqn:Huse; [ | discriminate ].
    injection Hfind as <- _ _.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (let_cont_handler_at_use_from run_valid scope tenv k params
             metadata handler handler' _ tenv_use args target'
             run_valid_kept Hleaf Huse);
      [ | | exact Hsat ].
    + intros vars' conts' statics' hp' rd' Hrun Hsat'.
      apply Hhandler; [ | exact Hsat' ].
      exact (run_valid_rest handler vars' conts' statics' _ _ Hrun).
    + apply rest_valid_run; [ | exact Hvalid ].
      cbn [sets_well_formed handler_sets_well_formed].
      rewrite Hhandler_sets. reflexivity.
Qed.

Theorem find_use_refines_from_valid : find_use_refines_from_valid_statement.
Proof.
  intros scope tenv k params metadata handler handler' body target xs ks
    Hfind Hfree Hsets Hhandler.
  cbn [sets_well_formed handler_sets_well_formed] in Hsets.
  apply andb_true_iff in Hsets. destruct Hsets as [Hhandler_sets Hbody_sets].
  apply refines_valid_from.
  apply (find_use_refines_valid body scope tenv k params metadata handler
           handler' target xs ks Hfind Hfree Hhandler_sets Hbody_sets).
  exact (from_valid_refines_valid target handler handler' Hhandler_sets
           Hhandler).
Qed.

(* ======================================================================== *)
(* The applications gathered                                                *)
(* ======================================================================== *)

(* What satisfies an environment in which two canonical elements have
   been made aliases satisfies the environment before. *)
Lemma alias_weakens (tenv : Typing_env.t) (e1 e2 : simple)
    (kind : flambda_kind) (result : Typing_env.add_alias_result)
    (statics : static_env) (valuation : env) :
  carried tenv ->
  alias_keeps_fit tenv e1 e2 kind = true ->
  Typing_env.add_alias tenv e1 e2 = Some (Ok result) ->
  typing_env_satisfied_strictly statics valuation
    (Typing_env.add_alias_result_t result) ->
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  intros [Hinvariant _] Hkeeps Hadd [Hbottom [Htypes Haliases]].
  unfold alias_keeps_fit in Hkeeps. rewrite !andb_true_iff in Hkeeps.
  destruct Hkeeps as [[[[[Hc1 Hc2] _] _] _] _].
  apply is_canonical_b_iff in Hc1. apply is_canonical_b_iff in Hc2.
  pose proof (proj1 (typing_env_well_formed_iff _) (proj1 Hinvariant))
    as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  destruct (add_alias_frame tenv e1 e2 result Hadd)
    as [[added [Hadded [Haliases' _]]]
        [_ [_ [_ [Hnames [_ [_ Hbottom']]]]]]].
  split; [ rewrite <- Hbottom'; exact Hbottom | ].
  split.
  - intros n ty binding_time_and_mode v Hlookup Hvalue.
    rewrite <- Hnames in Hlookup.
    exact (Htypes n ty binding_time_and_mode v Hlookup Hvalue).
  - rewrite Haliases' in Haliases.
    exact (add_respected_before _ _ _ _ e1 e2 added valuation Hconsistent
             Hc1 Hc2 Hadded Haliases).
Qed.

(* The environment of the body of a [Let]. *)
Lemma let_env_weakens (tenv tenv' : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) (statics : static_env)
    (valuation : env) :
  carried tenv ->
  let_env tenv x declared d = Some tenv' ->
  let_env_keeps_fit tenv x declared d = true ->
  typing_env_satisfied_strictly statics valuation tenv' ->
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  intros Hcarried. unfold let_env, let_env_keeps_fit. cbv zeta.
  generalize (resolve_value_slot tenv declared d). intros resolved.
  destruct (simplify_let tenv x resolved) as [ | d' tenv'' | ] eqn:Hlet;
    [ discriminate | | ].
  - intros [= <-]. revert Hlet. unfold simplify_let, let_keeps_fit.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [_ Hfresh].
    apply negb_true_iff in Hfresh.
    destruct (simplify_named tenv resolved) as [ | s | ty | kind | ];
      try discriminate.
    + unfold bind_simple.
      destruct (simplify_simple tenv s) as [[ty canonical] | ];
        [ | discriminate ].
      destruct (simple_in_terms_b tenv canonical
                && simple_of_kind_b tenv canonical
                     (FlambdaTypeGrammar.kind ty));
        [ | discriminate ].
      pose proof (add_variable_definition_carried tenv x
                    (FlambdaTypeGrammar.kind ty) Hcarried Hfresh)
        as Hcarried1.
      set (tenv1 :=
             Typing_env.add_variable_definition tenv x
               (FlambdaTypeGrammar.kind ty) Name_mode.normal) in *.
      destruct (Typing_env.add_alias tenv1 (Var x) canonical)
        as [[ | result | ] | ] eqn:Hadd; try discriminate.
      intros [= _ <-] Hkeeps Hsat.
      apply (add_variable_definition_satisfied_inv _ statics valuation tenv x
               (FlambdaTypeGrammar.kind ty) Name_mode.normal Hfresh).
      exact (alias_weakens tenv1 (Var x) canonical _ result statics valuation
               Hcarried1 Hkeeps Hadd Hsat).
    + destruct (type_in_scope_b tenv ty); [ | discriminate ].
      intros [= _ <-] _ Hsat.
      apply (add_variable_definition_satisfied_inv _ statics valuation tenv x
               (FlambdaTypeGrammar.kind ty) Name_mode.normal Hfresh).
      apply (replace_equation_satisfied_inv _ statics valuation _ (Name_var x)
               ty);
        [ | exact Hsat ].
      intros ty' binding_time_and_mode v Hlookup Hvalue.
      rewrite add_variable_definition_lookup in Hlookup.
      rewrite name_eqb_refl in Hlookup. injection Hlookup as <- _.
      right. apply unknown_satisfied.
      destruct Hsat as [_ [Htypes _]].
      pose proof (replace_equation_lookup
                    (Typing_env.add_variable_definition tenv x
                       (FlambdaTypeGrammar.kind ty) Name_mode.normal)
                    (Name_var x) ty)
        as Hnew.
      cbv beta iota in Hnew.
      rewrite add_variable_definition_lookup, name_eqb_refl in Hnew.
      cbn [option_map snd] in Hnew.
      exact (has_type_unless_kind _ statics valuation v ty
               (Htypes (Name_var x) ty _ v Hnew Hvalue)).
    + intros [= _ <-] _ Hsat.
      exact (add_variable_definition_satisfied_inv _ statics valuation tenv x
               kind Name_mode.normal Hfresh Hsat).
  - unfold declare_variable.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [_ Hfresh].
    apply negb_true_iff in Hfresh.
    intros [= <-] _ Hsat.
    exact (add_variable_definition_satisfied_inv _ statics valuation tenv x
             declared Name_mode.normal Hfresh Hsat).
Qed.

(* The environment of an arm. *)
Lemma arm_env_weakens (tenv : Typing_env.t) (scrutinee : simple) (d : Z)
    (statics : static_env) (valuation : env) :
  carried tenv ->
  arm_env_keeps_fit tenv scrutinee d = true ->
  typing_env_satisfied_strictly statics valuation (arm_env tenv scrutinee d)
  ->
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  intros Hcarried. unfold arm_env_keeps_fit, arm_env.
  destruct (simplify_simple tenv scrutinee) as [[ty [c | sym | const]] | ];
    try (intros _ Hsat; exact Hsat).
  destruct (simple_in_terms_b tenv (Var c)
            && simple_of_kind_b tenv (Var c) naked_immediate_kind);
    [ | intros _ Hsat; exact Hsat ].
  destruct (Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)))
    as [[ | result | ] | ] eqn:Hadd; try (intros _ Hsat; exact Hsat).
  intros Hkeeps Hsat.
  exact (alias_weakens tenv _ _ _ result statics valuation Hcarried Hkeeps
           Hadd Hsat).
Qed.

(* The environments of the applications at the end of a body. *)
Lemma leaf_uses_weakens (tenv : Typing_env.t) (k : continuation) (e : expr)
    (applications : list (Typing_env.t * list simple))
    (tenv_use : Typing_env.t) (args : list simple) (statics : static_env)
    (valuation : env) :
  carried tenv ->
  leaf_uses tenv k e = Some applications ->
  leaf_uses_keep_fit tenv k e = true ->
  (tenv_use, args) ∈ applications ->
  typing_env_satisfied_strictly statics valuation tenv_use ->
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  intros Hcarried.
  destruct e
    as [x declared d body | bindings body | k' h body | handlers body | ap
        | ac | scrutinee arms | ];
    simpl; try discriminate.
  - intros [= <-] _. destruct (Pos.eqb (ac_continuation ac) k);
      [ | intros [] ].
    intros [[= <- _] | []] Hsat. exact Hsat.
  - intros [= <-] Hkeeps Hin. apply in_flat_map in Hin.
    destruct Hin as [[d ac] [Harm Hin]]. cbn [fst snd] in Hin.
    rewrite forallb_forall in Hkeeps. specialize (Hkeeps (d, ac) Harm).
    cbn [fst snd] in Hkeeps.
    destruct (Pos.eqb (ac_continuation ac) k); [ | destruct Hin ].
    destruct Hin as [[= <- _] | []]. cbn [negb orb] in Hkeeps.
    exact (arm_env_weakens tenv scrutinee d statics valuation Hcarried
             Hkeeps).
Qed.

(* Variables defined. *)
Lemma define_variables_checked_weakens
    (definitions : list (variable * flambda_kind)) :
  ∀ (tenv tenv' : Typing_env.t) (statics : static_env) (valuation : env),
    define_variables_checked tenv definitions = Some tenv' ->
    typing_env_satisfied_strictly statics valuation tenv' ->
    typing_env_satisfied_strictly statics valuation tenv.
Proof.
  induction definitions as [ | [x kind] definitions IH];
    intros tenv tenv' statics valuation; cbn [define_variables_checked].
  - intros [= <-] Hsat. exact Hsat.
  - destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [_ Hfresh].
    apply negb_true_iff in Hfresh. intros Hdefine Hsat.
    apply (add_variable_definition_satisfied_inv _ statics valuation tenv x
             kind Name_mode.normal Hfresh).
    exact (IH _ tenv' statics valuation Hdefine Hsat).
Qed.

Lemma handler_env_weakens (tenv tenv_handler : Typing_env.t)
    (params : list variable) (metadata : handler_metadata)
    (statics : static_env) (valuation : env) :
  handler_env tenv params metadata = Some tenv_handler ->
  typing_env_satisfied_strictly statics valuation tenv_handler ->
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  unfold handler_env. cbv zeta.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | discriminate ].
  apply define_variables_checked_weakens.
Qed.

(* Every environment gathered has the two conditions, and what
   satisfies it satisfies the environment of the body. *)
Fixpoint gather_uses_carried_aux (body : expr) {struct body} :
  ∀ (tenv : Typing_env.t) (k : continuation) (h : continuation_handler)
    (uses : list (Typing_env.t * list simple)) (xs : list variable)
    (ks : list continuation),
    carried tenv ->
    gather_uses tenv k h body = Some (uses, xs, ks) ->
    gather_uses_keeps_fit tenv k h body = true ->
    ∀ use : Typing_env.t * list simple,
      use ∈ uses ->
      carried (fst use)
      ∧ ∀ (statics : static_env) (valuation : env),
          typing_env_satisfied_strictly statics valuation (fst use) ->
          typing_env_satisfied_strictly statics valuation tenv.
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros tenv k h uses xs ks Hcarried Hgather Hkeeps;
    cbn [gather_uses] in Hgather; cbn [gather_uses_keeps_fit] in Hkeeps;
    try discriminate.
  - destruct (var_free_in_handler_b x h); [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (gather_uses tenv' k h rest) as [[[uses' xs'] ks'] | ]
      eqn:Hrest; [ | discriminate ].
    injection Hgather as <- _ _.
    apply andb_true_iff in Hkeeps. destruct Hkeeps as [Hlet Hkeeps].
    assert (Hcarried' : carried tenv').
    { apply (let_env_known_to_fit tenv tenv' true x declared d
               (fun _ => Hcarried) Henv).
      rewrite Hlet. reflexivity. }
    intros use Hin.
    destruct (gather_uses_carried_aux rest tenv' k h uses' xs' ks' Hcarried'
                Hrest Hkeeps use Hin)
      as [Huse Hweakens].
    split; [ exact Huse | ].
    intros statics valuation Hsat.
    exact (let_env_weakens tenv tenv' x declared d statics valuation Hcarried
             Henv Hlet (Hweakens statics valuation Hsat)).
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k' h));
      [ | discriminate ].
    destruct (gather_uses tenv k h rest) as [[[uses_rest xs_rest] ks_rest] | ]
      eqn:Hrest; [ | discriminate ].
    apply andb_true_iff in Hkeeps. destruct Hkeeps as [Hkeeps_rest Hkeeps_h].
    pose proof (gather_uses_carried_aux rest tenv k h uses_rest xs_rest
                  ks_rest Hcarried Hrest Hkeeps_rest)
      as IHrest.
    destruct (cont_free_in_handler_b k h').
    + destruct h' as [params' metadata' body'].
      destruct (negb (cont_free_in_handler_b k h)
                && forallb (fun x => negb (var_free_in_handler_b x h))
                     params');
        [ | discriminate ].
      destruct (handler_env tenv params' metadata') as [tenv_handler | ]
        eqn:Henv; [ | discriminate ].
      destruct (gather_uses tenv_handler k h body')
        as [[[uses_h xs_h] ks_h] | ] eqn:Hinner; [ | discriminate ].
      injection Hgather as <- _ _.
      assert (Hcarried_h : carried tenv_handler).
      { exact (handler_env_known_to_fit tenv true params' metadata'
                 tenv_handler (fun _ => Hcarried) Henv eq_refl). }
      intros use Hin. apply in_app_or in Hin. destruct Hin as [Hin | Hin].
      * destruct (gather_uses_carried_aux body' tenv_handler k h uses_h xs_h
                    ks_h Hcarried_h Hinner Hkeeps_h use Hin)
          as [Huse Hweakens].
        split; [ exact Huse | ].
        intros statics valuation Hsat.
        exact (handler_env_weakens tenv tenv_handler params' metadata'
                 statics valuation Henv (Hweakens statics valuation Hsat)).
      * exact (IHrest use Hin).
    + injection Hgather as <- _ _. exact IHrest.
  - destruct (leaf_uses tenv k (Apply_cont ac)) as [uses' | ] eqn:Hleaf;
      [ | discriminate ].
    injection Hgather as <- _ _. intros [tenv_use args] Hin. cbn [fst].
    split.
    + exact (leaf_uses_carried tenv k _ uses' tenv_use args Hcarried Hleaf
               Hkeeps Hin).
    + intros statics valuation.
      exact (leaf_uses_weakens tenv k _ uses' tenv_use args statics valuation
               Hcarried Hleaf Hkeeps Hin).
  - destruct (leaf_uses tenv k (Switch scrutinee arms)) as [uses' | ]
      eqn:Hleaf; [ | discriminate ].
    injection Hgather as <- _ _. intros [tenv_use args] Hin. cbn [fst].
    split.
    + exact (leaf_uses_carried tenv k _ uses' tenv_use args Hcarried Hleaf
               Hkeeps Hin).
    + intros statics valuation.
      exact (leaf_uses_weakens tenv k _ uses' tenv_use args statics valuation
               Hcarried Hleaf Hkeeps Hin).
Qed.

Theorem gather_uses_known_to_fit : gather_uses_known_to_fit_statement.
Proof.
  intros tenv fit k h body uses xs ks use Hknown Hgather Hin Hfit.
  apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
  exact (proj1
           (gather_uses_carried_aux body tenv k h uses xs ks (Hknown Hfit)
              Hgather Hkeeps use Hin)).
Qed.

Theorem gather_uses_weaken : gather_uses_weaken_statement.
Proof.
  intros tenv k h body uses xs ks use statics valuation Hknown Hgather Hkeeps
    Hin.
  exact (proj2
           (gather_uses_carried_aux body tenv k h uses xs ks
              (Hknown eq_refl) Hgather Hkeeps use Hin)
           statics valuation).
Qed.

(* FlambdaSimplifyPassProofs.gather_uses_refines_aux, from the states
   that have [run_valid]. Where the handler of another continuation
   applies the continuation, the copy of the handler as rewritten is
   run in that handler: its sets are to be well formed. *)
Fixpoint gather_uses_refines_valid (body : expr) {struct body} :
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' : expr)
    (target : Typing_env.t) (uses : list (Typing_env.t * list simple))
    (xs : list variable) (ks : list continuation),
    gather_uses tenv k (Handler params metadata handler) body
    = Some (uses, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    sets_well_formed handler = true ->
    sets_well_formed handler' = true ->
    sets_well_formed body = true ->
    (∀ use : Typing_env.t * list simple,
      use ∈ uses ->
      application_admits params metadata (fst use) (snd use) target) ->
    refines_valid target handler handler' ->
    refines_valid tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros tenv k params metadata handler handler' target uses xs ks Hgather
      Hfree Hhandler_sets Hhandler_sets' Hbody_sets Huses Hhandler;
    cbn [gather_uses] in Hgather; try discriminate.
  - destruct (var_free_in_handler_b x (Handler params metadata handler))
      eqn:Hx; [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (gather_uses tenv' k (Handler params metadata handler) rest)
      as [[[uses' xs'] ks'] | ] eqn:Hrest; [ | discriminate ].
    injection Hgather as <- <- <-.
    unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
    apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
    apply andb_true_iff in Hxs. destruct Hxs as [Hx' Hxs].
    apply negb_true_iff in Hx'.
    assert (Hrest_free :
      binders_not_free xs' ks' (Handler params metadata handler') = true).
    { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
    cbn [sets_well_formed] in Hbody_sets.
    apply andb_true_iff in Hbody_sets. destruct Hbody_sets as [Hd Hrest_sets].
    pose proof (gather_uses_refines_valid rest tenv' k params metadata
                  handler handler' target uses' xs' ks' Hrest Hrest_free
                  Hhandler_sets Hhandler_sets' Hrest_sets Huses Hhandler)
      as IH.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest))
                vars conts statics hp rd)).
    { apply equivalent_refines, let_cont_past_let. intros Hfree'.
      apply var_free_in_handler_b_sound in Hfree'. congruence. }
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler') rest))
                vars conts statics hp rd)).
    { apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
      apply IH.
      - refine
          (rest_valid_let x declared d
             (Let_cont_non_recursive k (Handler params metadata handler)
                rest)
             vars conts statics hp hp' rd rd' v Hd _ Hvalid Hv Hkind).
        cbn [sets_well_formed handler_sets_well_formed].
        rewrite Hhandler_sets, Hrest_sets. reflexivity.
      - exact (let_env_satisfied tenv tenv' x declared d statics hp hp' rd
                 rd' vars v Henv Hsat Hv Hkind). }
    apply equivalent_refines, equivalent_sym, let_cont_past_let.
    intros Hfree'. apply var_free_in_handler_b_sound in Hfree'. congruence.
  - destruct (negb (Pos.eqb k k')
              && negb
                   (cont_free_in_handler_b k'
                      (Handler params metadata handler)))
      eqn:Htests; [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hne Hk'].
    apply negb_true_iff in Hne, Hk'. apply Pos.eqb_neq in Hne.
    destruct (gather_uses tenv k (Handler params metadata handler) rest)
      as [[[uses_rest xs_rest] ks_rest] | ] eqn:Hrest; [ | discriminate ].
    cbn [sets_well_formed] in Hbody_sets.
    apply andb_true_iff in Hbody_sets.
    destruct Hbody_sets as [Hh'_sets Hrest_sets].
    destruct (cont_free_in_handler_b k h') eqn:Hk.
    + (* the handler applies the continuation: a copy goes into it *)
      destruct h' as [params' metadata' body'].
      cbn [handler_sets_well_formed] in Hh'_sets.
      destruct (negb
                  (cont_free_in_handler_b k
                     (Handler params metadata handler))
                && forallb
                     (fun x =>
                        negb
                          (var_free_in_handler_b x
                             (Handler params metadata handler)))
                     params')
        eqn:Htests'; [ | discriminate ].
      apply andb_true_iff in Htests'. destruct Htests' as [Hself Hparams].
      apply negb_true_iff in Hself.
      destruct (handler_env tenv params' metadata') as [tenv_handler | ]
        eqn:Henv; [ | discriminate ].
      destruct (gather_uses tenv_handler k (Handler params metadata handler)
                  body')
        as [[[uses_h xs_h] ks_h] | ] eqn:Hinner; [ | discriminate ].
      injection Hgather as <- <- <-.
      change (k :: k' :: ks_h ++ ks_rest)
        with ([k; k'] ++ ks_h ++ ks_rest) in Hfree.
      rewrite binders_not_free_app, binders_not_free_app in Hfree.
      apply andb_true_iff in Hfree. destruct Hfree as [Hfree_head Hfree].
      apply andb_true_iff in Hfree. destruct Hfree as [Hfree_h Hfree_rest].
      unfold binders_not_free in Hfree_head. cbn [forallb] in Hfree_head.
      apply andb_true_iff in Hfree_head.
      destruct Hfree_head as [Hparams' Hconts'].
      rewrite andb_true_r in Hconts'.
      apply andb_true_iff in Hconts'. destruct Hconts' as [Hself' Hk''].
      apply negb_true_iff in Hself', Hk''.
      pose proof (gather_uses_refines_valid body' tenv_handler k params
                    metadata handler handler' target uses_h xs_h ks_h Hinner
                    Hfree_h Hhandler_sets Hhandler_sets' Hh'_sets
                    (fun use Hin =>
                       Huses use (in_or_app _ _ _ (or_introl Hin)))
                    Hhandler)
        as IHh.
      pose proof (gather_uses_refines_valid rest tenv k params metadata
                    handler handler' target uses_rest xs_rest ks_rest Hrest
                    Hfree_rest Hhandler_sets Hhandler_sets' Hrest_sets
                    (fun use Hin =>
                       Huses use (in_or_app _ _ _ (or_intror Hin)))
                    Hhandler)
        as IHrest.
      set (h := Handler params metadata handler) in *.
      set (hr := Handler params metadata handler') in *.
      assert (Hh_sets : handler_sets_well_formed h = true).
      { exact Hhandler_sets. }
      assert (Hhr_sets : handler_sets_well_formed hr = true).
      { exact Hhandler_sets'. }
      assert (Hnot_self : ~ cont_free_in_handler k h).
      { intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      assert (Hnot_self' : ~ cont_free_in_handler k hr).
      { intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      assert (Hnot_params :
        ∀ param : variable, param ∈ params' -> ~ var_free_in_handler param h).
      { intros param Hin Hf. apply var_free_in_handler_b_sound in Hf.
        rewrite forallb_forall in Hparams. specialize (Hparams param Hin).
        rewrite Hf in Hparams. discriminate. }
      assert (Hnot_params' :
        ∀ param : variable,
          param ∈ params' -> ~ var_free_in_handler param hr).
      { intros param Hin Hf. apply var_free_in_handler_b_sound in Hf.
        rewrite forallb_forall in Hparams'. specialize (Hparams' param Hin).
        rewrite Hf in Hparams'. discriminate. }
      intros vars conts statics hp rd Hvalid Hsat.
      (* the copy *)
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k h
                     (Let_cont_non_recursive k'
                        (Handler params' metadata'
                           (Let_cont_non_recursive k h body'))
                        rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines.
        exact (let_cont_copied_into_handler k k' h params' metadata' body'
                 rest Hnot_self Hnot_params vars conts statics hp rd). }
      (* the copy rewritten, in the handler *)
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k h
                     (Let_cont_non_recursive k'
                        (Handler params' metadata'
                           (Let_cont_non_recursive k hr body'))
                        rest))
                  vars conts statics hp rd)).
      { apply (let_cont_body_refines_valid tenv k h _ _);
          [ | | exact Hvalid | exact Hsat ].
        - cbn [sets_well_formed handler_sets_well_formed].
          rewrite Hh_sets, Hh'_sets, Hrest_sets. reflexivity.
        - apply (let_cont_handler_typed_valid tenv k' params' metadata' _ _
                   rest tenv_handler Henv);
            [ | | exact Hrest_sets | exact IHh ].
          + cbn [sets_well_formed]. rewrite Hh_sets, Hh'_sets. reflexivity.
          + cbn [sets_well_formed]. rewrite Hhr_sets, Hh'_sets. reflexivity. }
      (* the original past the other [Let_cont], which no longer uses it *)
      assert (Hbound :
        ~ cont_free_in_handler k
            (Handler params' metadata' (Let_cont_non_recursive k hr body'))).
      { simpl. tauto. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k'
                     (Handler params' metadata'
                        (Let_cont_non_recursive k hr body'))
                     (Let_cont_non_recursive k h rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines, let_cont_past_let_cont;
          [ exact Hne | exact Hbound | ].
        intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k'
                     (Handler params' metadata'
                        (Let_cont_non_recursive k hr body'))
                     (Let_cont_non_recursive k hr rest))
                  vars conts statics hp rd)).
      { apply (let_cont_body_refines_valid tenv k' _ _ _);
          [ | exact IHrest | exact Hvalid | exact Hsat ].
        cbn [sets_well_formed handler_sets_well_formed].
        rewrite Hhr_sets, Hh'_sets, Hh_sets, Hrest_sets. reflexivity. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k hr
                     (Let_cont_non_recursive k'
                        (Handler params' metadata'
                           (Let_cont_non_recursive k hr body'))
                        rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines, equivalent_sym, let_cont_past_let_cont;
          [ exact Hne | exact Hbound | ].
        intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      apply equivalent_refines, equivalent_sym.
      exact (let_cont_copied_into_handler k k' hr params' metadata' body'
               rest Hnot_self' Hnot_params' vars conts statics hp rd).
    + injection Hgather as <- <- <-.
      unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
      apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
      apply andb_true_iff in Hks. destruct Hks as [Hk'' Hks].
      apply negb_true_iff in Hk''.
      assert (Hrest_free :
        binders_not_free xs_rest ks_rest (Handler params metadata handler')
        = true).
      { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
      pose proof (gather_uses_refines_valid rest tenv k params metadata
                    handler handler' target uses_rest xs_rest ks_rest Hrest
                    Hrest_free Hhandler_sets Hhandler_sets' Hrest_sets Huses
                    Hhandler)
        as IH.
      intros vars conts statics hp rd Hvalid Hsat.
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k' h'
                     (Let_cont_non_recursive k
                        (Handler params metadata handler) rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines, let_cont_past_let_cont;
          [ exact Hne | | ];
          intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
          congruence. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k' h'
                     (Let_cont_non_recursive k
                        (Handler params metadata handler') rest))
                  vars conts statics hp rd)).
      { apply (let_cont_body_refines_valid tenv k' h' _ _);
          [ | exact IH | exact Hvalid | exact Hsat ].
        cbn [sets_well_formed handler_sets_well_formed].
        rewrite Hh'_sets, Hhandler_sets, Hrest_sets. reflexivity. }
      apply equivalent_refines, equivalent_sym, let_cont_past_let_cont;
        [ exact Hne | | ];
        intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
        congruence.
  - destruct (leaf_uses tenv k (Apply_cont ac)) as [uses' | ] eqn:Hleaf;
      [ | discriminate ].
    injection Hgather as <- _ _.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (leaf_refines_from run_valid tenv k params metadata handler
             handler' _ uses' run_valid_kept Hleaf);
      [ | | exact Hsat ].
    + intros index tenv_use args statics' hp' rd' valuation vars' conts' vs
        handler_vars Hnth _ Hshows Hargs Hkinds Hbind Hrun.
      apply Hhandler.
      * exact (run_valid_rest handler handler_vars conts' statics' _ _ Hrun).
      * exact (Huses (tenv_use, args) (nth_error_In _ _ Hnth) statics' vars'
                 vs handler_vars
                 (shows_satisfies tenv_use statics' valuation vars' Hshows)
                 Hargs Hkinds Hbind).
    + apply rest_valid_run; [ | exact Hvalid ].
      cbn [sets_well_formed handler_sets_well_formed].
      rewrite Hhandler_sets. reflexivity.
  - destruct (leaf_uses tenv k (Switch scrutinee arms)) as [uses' | ]
      eqn:Hleaf; [ | discriminate ].
    injection Hgather as <- _ _.
    intros vars conts statics hp rd Hvalid Hsat.
    apply (leaf_refines_from run_valid tenv k params metadata handler
             handler' _ uses' run_valid_kept Hleaf);
      [ | | exact Hsat ].
    + intros index tenv_use args statics' hp' rd' valuation vars' conts' vs
        handler_vars Hnth _ Hshows Hargs Hkinds Hbind Hrun.
      apply Hhandler.
      * exact (run_valid_rest handler handler_vars conts' statics' _ _ Hrun).
      * exact (Huses (tenv_use, args) (nth_error_In _ _ Hnth) statics' vars'
                 vs handler_vars
                 (shows_satisfies tenv_use statics' valuation vars' Hshows)
                 Hargs Hkinds Hbind).
    + apply rest_valid_run; [ | exact Hvalid ].
      cbn [sets_well_formed handler_sets_well_formed].
      rewrite Hhandler_sets. reflexivity.
Qed.

Theorem gather_uses_refines_from_valid
    : gather_uses_refines_from_valid_statement.
Proof.
  intros tenv k params metadata handler handler' body target uses xs ks
    Hgather Hfree Hsets Hhandler_sets' Huses Hhandler.
  cbn [sets_well_formed handler_sets_well_formed] in Hsets.
  apply andb_true_iff in Hsets. destruct Hsets as [Hhandler_sets Hbody_sets].
  apply refines_valid_from.
  apply (gather_uses_refines_valid body tenv k params metadata handler
           handler' target uses xs ks Hgather Hfree Hhandler_sets
           Hhandler_sets' Hbody_sets Huses).
  exact (from_valid_refines_valid target handler handler' Hhandler_sets
           Hhandler).
Qed.

(* The one application gathered, and [find_one_use]. *)
Lemma find_gathered_use_known_to_fit (tenv : Typing_env.t) (fit : bool)
    (k : continuation) (h : continuation_handler) (body : expr)
    (target : Typing_env.t) (xs : list variable) (ks : list continuation) :
  known_to_fit tenv fit ->
  find_gathered_use tenv k h body = Some (target, xs, ks) ->
  known_to_fit target (fit && find_gathered_use_keeps_fit tenv k h body).
Proof.
  intros Hknown Hfind Hfit.
  apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
  destruct h as [params metadata handler].
  unfold find_gathered_use in Hfind.
  unfold find_gathered_use_keeps_fit in Hkeeps.
  destruct (gather_uses tenv k (Handler params metadata handler) body)
    as [[[uses xs'] ks'] | ] eqn:Hgather; [ | discriminate ].
  destruct uses as [ | [tenv_use args] [ | second others]];
    try discriminate.
  destruct (use_env tenv tenv_use params metadata args) as [target' | ]
    eqn:Huse; [ | discriminate ].
  injection Hfind as <- _ _.
  apply andb_true_iff in Hkeeps. destruct Hkeeps as [Hgathered Hused].
  apply (use_env_carried tenv tenv_use params metadata args target');
    [ | exact Huse | exact Hused ].
  exact (proj1
           (gather_uses_carried_aux body tenv k _ _ xs' ks' (Hknown Hfit)
              Hgather Hgathered (tenv_use, args) (or_introl eq_refl))).
Qed.

Theorem find_one_use_known_to_fit : find_one_use_known_to_fit_statement.
Proof.
  intros tenv fit k h body target xs ks Hknown.
  unfold find_one_use, find_one_use_keeps_fit.
  destruct (find_use tenv tenv k h body) as [found | ] eqn:Huse.
  - intros Hfind. injection Hfind as ->.
    exact (find_use_known_to_fit_aux body tenv tenv fit k h target xs ks
             Hknown Huse).
  - exact (find_gathered_use_known_to_fit tenv fit k h body target xs ks
             Hknown).
Qed.

Lemma find_one_use_refines_valid (tenv : Typing_env.t) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (target : Typing_env.t)
    (xs : list variable) (ks : list continuation) :
  find_one_use tenv k (Handler params metadata handler) body
  = Some (target, xs, ks) ->
  binders_not_free xs ks (Handler params metadata handler') = true ->
  sets_well_formed handler = true ->
  sets_well_formed handler' = true ->
  sets_well_formed body = true ->
  refines_valid target handler handler' ->
  refines_valid tenv
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  unfold find_one_use.
  intros Hfind Hfree Hhandler_sets Hhandler_sets' Hbody_sets Hhandler.
  destruct (find_use tenv tenv k (Handler params metadata handler) body)
    as [found | ] eqn:Huse.
  - injection Hfind as ->.
    exact (find_use_refines_valid body tenv tenv k params metadata handler
             handler' target xs ks Huse Hfree Hhandler_sets Hbody_sets
             Hhandler).
  - unfold find_gathered_use in Hfind.
    destruct (gather_uses tenv k (Handler params metadata handler) body)
      as [[[uses xs'] ks'] | ] eqn:Hgather; [ | discriminate ].
    destruct uses as [ | [tenv_use args] [ | second others]];
      try discriminate.
    destruct (use_env tenv tenv_use params metadata args) as [target' | ]
      eqn:Hmade; [ | discriminate ].
    injection Hfind as <- <- <-.
    apply (gather_uses_refines_valid body tenv k params metadata handler
             handler' target' [(tenv_use, args)] xs' ks' Hgather Hfree
             Hhandler_sets Hhandler_sets' Hbody_sets);
      [ | exact Hhandler ].
    intros use [<- | []]. cbn [fst snd].
    intros statics vars vs handler_vars Hsat Hargs Hkinds Hbind.
    exact (use_env_satisfied tenv tenv_use params metadata args target'
             statics vars vs handler_vars Hmade Hsat Hargs Hkinds Hbind).
Qed.

(* ======================================================================== *)
(* A term                                                                   *)
(* ======================================================================== *)

(* By recursion on the term, as for the first function: the bodies of
   the code that a [Let_static] binds are rewritten one binding at a
   time, each by the statement of FlambdaContextWhereSpec.v
   for the body of code at a predicate that a step keeps. The two
   whole states have [run_valid] together because the sets of the two
   groups are well formed together. *)
Fixpoint simplify_expr_pruning_sound (prune : switch_rewrite)
    (Hprune : switch_rewrite_sound prune) (e : expr) {struct e} :
  ∀ (tenv : Typing_env.t) (fit : bool),
    known_to_fit tenv fit ->
    sets_well_formed e = true ->
    sets_well_formed (simplify_expr_pruning prune tenv fit e) = true
    ∧ refines_valid tenv e (simplify_expr_pruning prune tenv fit e).
Proof.
  destruct e
    as [x declared d body | bindings body | k handler body
        | handlers body | ap | ac | scrutinee arms | ];
    intros tenv fit Hknown Hsets; cbn [simplify_expr_pruning].
  - pose proof Hsets as H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H. destruct H as [Hd Hbody].
    pose proof (resolve_value_slot_refines tenv x declared d body)
      as Hresolve.
    pose proof (resolve_value_slot_sets tenv declared d Hd) as Hd0.
    cbv zeta.
    set (d0 := resolve_value_slot tenv declared d) in *.
    pose proof (simplify_let_sound tenv x d0) as Hsound.
    destruct (simplify_let tenv x d0) as [ | d' tenv' | ] eqn:Hlet.
    + split; [ reflexivity | ].
      apply refines_under_valid.
      eapply expr_refines_under_trans; [ exact Hresolve | ].
      apply expr_equivalent_under_refines. exact (Hsound declared body).
    + destruct (simplify_let_valid tenv tenv' x d0 d' Hlet Hd0)
        as [Hd' Hstep].
      destruct (simplify_expr_pruning_sound prune Hprune body tenv'
                  (fit && let_keeps_fit tenv x d0)
                  (let_keeps_fit_sound tenv tenv' fit x d0 d' Hknown Hlet)
                  Hbody)
        as [Hbody' Hrefines].
      split.
      * cbn [sets_well_formed]. rewrite Hd', Hbody'. reflexivity.
      * eapply refines_valid_trans;
          [ apply refines_under_valid; exact Hresolve | ].
        exact (Hstep declared body _ Hbody Hrefines).
    + destruct (declare_variable tenv x declared) as [tenv' | ] eqn:Hdeclare.
      * revert Hdeclare. unfold declare_variable.
        destruct (typing_env_well_formed tenv
                  && negb (Typing_env.mem None tenv (Name_var x)))
          eqn:Htests; [ | discriminate ].
        intros [= <-]. apply andb_true_iff in Htests.
        destruct Htests as [Hwf Hfresh]. apply negb_true_iff in Hfresh.
        destruct (simplify_expr_pruning_sound prune Hprune body _ fit
                    (fun Hfit =>
                       add_variable_definition_carried tenv x declared
                         (Hknown Hfit) Hfresh)
                    Hbody)
          as [Hbody' Hrefines].
        split.
        -- cbn [sets_well_formed]. rewrite Hd, Hbody'. reflexivity.
        -- exact (let_declared_refines_valid tenv x declared d body _ Hwf
                    Hfresh Hd Hbody Hrefines).
      * split; [ exact Hsets | ].
        apply refines_under_valid, expr_refines_under_refl.
  - assert (Hbody : sets_well_formed body = true).
    { pose proof Hsets as H. cbn [sets_well_formed] in H.
      apply andb_true_iff in H. exact (proj2 H). }
    cbv zeta.
    assert (Hbindings :
      ∀ (tenv' : Typing_env.t) (fit' : bool) (before : list static_binding),
        known_to_fit tenv' fit' ->
        add_static_bindings tenv (before ++ bindings) = tenv' ->
        sets_well_formed (Let_static (before ++ bindings) body) = true ->
        sets_well_formed
          (Let_static
             (before
              ++ map (simplify_static_binding_pruning prune tenv' fit')
                   bindings)
             body)
        = true
        ∧ refines_valid tenv (Let_static (before ++ bindings) body)
            (Let_static
               (before
                ++ map (simplify_static_binding_pruning prune tenv' fit')
                     bindings)
               body)).
    { clear Hsets. intros tenv' fit' before Hknown'. revert before.
      induction bindings as [ | binding bindings IH];
        intros before Henv Hsets.
      { split; [ exact Hsets | ].
        apply refines_under_valid, expr_refines_under_refl. }
      cbn [map].
      assert (Hone :
        sets_well_formed
          (Let_static
             (before
              ++ simplify_static_binding_pruning prune tenv' fit' binding
                 :: bindings)
             body)
        = true
        ∧ refines_valid tenv (Let_static (before ++ binding :: bindings) body)
            (Let_static
               (before
                ++ simplify_static_binding_pruning prune tenv' fit' binding
                   :: bindings)
               body)).
      { assert (Hsame :
          sets_well_formed (Let_static (before ++ binding :: bindings) body)
          = true
          ∧ refines_valid tenv
              (Let_static (before ++ binding :: bindings) body)
              (Let_static (before ++ binding :: bindings) body)).
        { split; [ exact Hsets | ].
          apply refines_under_valid, expr_refines_under_refl. }
        destruct binding as [id c | id | sym const];
          cbn [simplify_static_binding_pruning]; try exact Hsame.
        destruct c as [params my_closure ret exn metadata code_body].
        cbn [simplify_code_pruning].
        destruct (code_env_in tenv' params my_closure metadata)
          as [tenv_code | ] eqn:Hcode_env; [ | exact Hsame ].
        cbv zeta.
        destruct (binds_statics code_body) eqn:Hsource; cbn [negb andb];
          [ exact Hsame | ].
        destruct (binds_statics
                    (simplify_expr_pruning prune tenv_code fit' code_body))
          eqn:Htarget; cbn [negb];
          [ exact Hsame | ].
        clear Hsame.
        assert (Hparts :
          forallb binding_sets_well_formed before = true
          ∧ sets_well_formed code_body = true
          ∧ forallb binding_sets_well_formed bindings = true).
        { pose proof Hsets as H. cbn [sets_well_formed] in H.
          rewrite !andb_true_iff in H. destruct H as [[_ H] _].
          rewrite forallb_app in H. apply andb_true_iff in H.
          destruct H as [Hbefore H].
          change (sets_well_formed code_body
                  && forallb binding_sets_well_formed bindings
                  = true) in H.
          apply andb_true_iff in H. exact (conj Hbefore H). }
        destruct Hparts as [Hbefore [Hcode Hrest]].
        destruct (simplify_expr_pruning_sound prune Hprune code_body
                    tenv_code fit'
                    (code_env_in_known_to_fit tenv' fit' params my_closure
                       metadata tenv_code Hknown' Hcode_env)
                    Hcode)
          as [Hcode' Hcode_refines].
        assert (Hsets' :
          sets_well_formed
            (Let_static
               (before
                ++ Bind_code id
                     (Code params my_closure ret exn metadata
                        (simplify_expr_pruning prune tenv_code fit'
                           code_body))
                   :: bindings)
               body)
          = true).
        { apply (let_static_sets_alike
                   (before
                    ++ Bind_code id
                         (Code params my_closure ret exn metadata code_body)
                       :: bindings)
                   _ body body);
            [ rewrite !map_app; reflexivity | | exact Hbody | exact Hsets ].
          rewrite forallb_app, Hbefore.
          change (sets_well_formed
                    (simplify_expr_pruning prune tenv_code fit' code_body)
                  && forallb binding_sets_well_formed bindings
                  = true).
          rewrite Hcode', Hrest. reflexivity. }
        split; [ exact Hsets' | ].
        apply (let_static_code_typed_valid tenv before id params my_closure
                 ret exn metadata code_body _ bindings body tenv_code);
          [ rewrite Henv; exact Hcode_env | exact Hsource | exact Htarget
          | exact Hcode | exact Hcode' | exact Hsets | exact Hsets'
          | exact Hcode_refines ]. }
      destruct Hone as [Hsets1 Hrefines1].
      specialize
        (IH
           (before
            ++ [simplify_static_binding_pruning prune tenv' fit' binding])).
      rewrite <- !app_assoc in IH. cbn [app] in IH.
      assert (Henv1 :
        add_static_bindings tenv
          (before
           ++ simplify_static_binding_pruning prune tenv' fit' binding
              :: bindings)
        = tenv').
      { rewrite <- Henv.
        destruct binding as [id c | id | sym const];
          cbn [simplify_static_binding_pruning]; try reflexivity.
        apply add_static_bindings_code. }
      destruct (IH Henv1 Hsets1) as [Hsets2 Hrefines2].
      split; [ exact Hsets2 | ].
      exact (refines_valid_trans tenv _ _ _ Hrefines1 Hrefines2). }
    destruct (Hbindings (add_static_bindings tenv bindings)
                (fit && statics_keep_fit tenv bindings) []
                (statics_keep_fit_sound tenv fit bindings Hknown) eq_refl
                Hsets)
      as [Hsets1 Hrefines1].
    cbn [app] in Hsets1, Hrefines1.
    destruct (simplify_expr_pruning_sound prune Hprune body _ _
                (statics_keep_fit_sound tenv fit bindings Hknown) Hbody)
      as [Hbody' Hbody_refines].
    split.
    + pose proof Hsets1 as H. cbn [sets_well_formed] in H |- *.
      rewrite !andb_true_iff in H |- *. destruct H as [H _].
      exact (conj H Hbody').
    + eapply refines_valid_trans; [ exact Hrefines1 | ].
      apply let_static_body_refines_valid; [ exact Hsets1 | ].
      rewrite add_static_bindings_simplify_pruning. exact Hbody_refines.
  - pose proof Hsets as H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H. destruct H as [Hhandler Hbody].
    destruct (simplify_expr_pruning_sound prune Hprune body tenv fit Hknown
                Hbody)
      as [Hbody' Hrefines].
    pose proof (let_cont_body_refines_valid tenv k handler body _ Hsets
                  Hrefines)
      as Hfirst.
    destruct handler as [params metadata handler].
    cbn [handler_sets_well_formed] in Hhandler. cbv zeta.
    revert Hbody' Hrefines Hfirst.
    generalize (simplify_expr_pruning prune tenv fit body).
    intros body' Hbody' Hrefines Hfirst.
    assert (Hdefault :
      sets_well_formed
        (Let_cont_non_recursive k
           (simplify_handler_pruning prune false tenv fit
              (Handler params metadata handler))
           body')
      = true
      ∧ refines_valid tenv
          (Let_cont_non_recursive k (Handler params metadata handler) body)
          (Let_cont_non_recursive k
             (simplify_handler_pruning prune false tenv fit
                (Handler params metadata handler))
             body')).
    { cbn [simplify_handler_pruning].
      destruct (handler_env tenv params metadata) as [tenv_handler | ]
        eqn:Henv.
      + cbv zeta. cbn [negb orb].
        destruct (simplify_expr_pruning_sound prune Hprune handler
                    tenv_handler fit
                    (handler_env_known_to_fit tenv fit params metadata
                       tenv_handler Hknown Henv)
                    Hhandler)
          as [Hhandler' Hhandler_refines].
        split.
        * cbn [sets_well_formed handler_sets_well_formed].
          rewrite Hhandler', Hbody'. reflexivity.
        * eapply refines_valid_trans; [ exact Hfirst | ].
          exact (let_cont_handler_typed_valid tenv k params metadata handler
                   _ _ tenv_handler Henv Hhandler Hhandler' Hbody'
                   Hhandler_refines).
      + split; [ | exact Hfirst ].
        cbn [sets_well_formed handler_sets_well_formed].
        rewrite Hhandler, Hbody'. reflexivity. }
    destruct (find_one_use tenv k (Handler params metadata handler) body')
      as [[[target xs] ks] | ] eqn:Hfind; [ | exact Hdefault ].
    destruct (simplify_expr_pruning_sound prune Hprune handler target
                (fit
                 && find_one_use_keeps_fit tenv k
                      (Handler params metadata handler) body')
                (find_one_use_known_to_fit tenv fit k _ body' target xs ks
                   Hknown Hfind)
                Hhandler)
      as [Hhandler' Hhandler_refines].
    destruct (binders_not_free xs ks
                (Handler params metadata
                   (simplify_expr_pruning prune target
                      (fit
                       && find_one_use_keeps_fit tenv k
                            (Handler params metadata handler) body')
                      handler)))
      eqn:Hfree; [ | exact Hdefault ].
    split.
    + cbn [sets_well_formed handler_sets_well_formed].
      rewrite Hhandler', Hbody'. reflexivity.
    + eapply refines_valid_trans; [ exact Hfirst | ].
      exact (find_one_use_refines_valid tenv k params metadata handler _
               body' target xs ks Hfind Hfree Hhandler Hhandler' Hbody'
               Hhandler_refines).
  - pose proof Hsets as H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H. destruct H as [Hhandlers Hbody].
    destruct (simplify_expr_pruning_sound prune Hprune body tenv fit Hknown
                Hbody)
      as [Hbody' Hrefines].
    pose proof (let_cont_recursive_body_refines_valid tenv handlers body _
                  Hsets Hrefines)
      as Hfirst.
    assert (Hsets1 :
      sets_well_formed
        (Let_cont_recursive handlers
           (simplify_expr_pruning prune tenv fit body))
      = true).
    { cbn [sets_well_formed]. rewrite Hhandlers, Hbody'. reflexivity. }
    revert Hfirst Hsets1.
    generalize (simplify_expr_pruning prune tenv fit body).
    intros body' Hfirst Hsets1.
    assert (Hall :
      ∀ before : list (continuation * continuation_handler),
        sets_well_formed (Let_cont_recursive (before ++ handlers) body')
        = true ->
        sets_well_formed
          (Let_cont_recursive
             (before
              ++ map
                   (fun '(k, h) =>
                      (k, simplify_handler_pruning prune true tenv fit h))
                   handlers)
             body')
        = true
        ∧ refines_valid tenv
            (Let_cont_recursive (before ++ handlers) body')
            (Let_cont_recursive
               (before
                ++ map
                     (fun '(k, h) =>
                        (k, simplify_handler_pruning prune true tenv fit h))
                     handlers)
               body')).
    { clear Hsets Hhandlers Hfirst Hsets1 Hrefines.
      induction handlers as [ | [k h] handlers IH]; intros before Hall.
      { split; [ exact Hall | ].
        apply refines_under_valid, expr_refines_under_refl. }
      cbn [map].
      assert (Hone :
        sets_well_formed
          (Let_cont_recursive
             (before
              ++ (k, simplify_handler_pruning prune true tenv fit h)
                 :: handlers)
             body')
        = true
        ∧ refines_valid tenv
            (Let_cont_recursive (before ++ (k, h) :: handlers) body')
            (Let_cont_recursive
               (before
                ++ (k, simplify_handler_pruning prune true tenv fit h)
                   :: handlers)
               body')).
      { assert (Hsame :
          sets_well_formed
            (Let_cont_recursive (before ++ (k, h) :: handlers) body')
          = true
          ∧ refines_valid tenv
              (Let_cont_recursive (before ++ (k, h) :: handlers) body')
              (Let_cont_recursive (before ++ (k, h) :: handlers) body')).
        { split; [ exact Hall | ].
          apply refines_under_valid, expr_refines_under_refl. }
        destruct h as [params metadata handler].
        cbn [simplify_handler_pruning].
        destruct (handler_env tenv params metadata) as [tenv_handler | ]
          eqn:Henv; [ | exact Hsame ].
        cbv zeta. cbn [negb orb].
        destruct (Bool.eqb (binds_statics handler)
                    (binds_statics
                       (simplify_expr_pruning prune tenv_handler fit
                          handler)))
          eqn:Hbinds; [ | exact Hsame ].
        clear Hsame.
        pose proof Hall as Hparts.
        rewrite let_cont_recursive_sets in Hparts.
        apply andb_true_iff in Hparts.
        destruct Hparts as [Hhandler Hothers].
        cbn [handler_sets_well_formed] in Hhandler.
        destruct (simplify_expr_pruning_sound prune Hprune handler
                    tenv_handler fit
                    (handler_env_known_to_fit tenv fit params metadata
                       tenv_handler Hknown Henv)
                    Hhandler)
          as [Hhandler' Hhandler_refines].
        assert (Hall' :
          sets_well_formed
            (Let_cont_recursive
               (before
                ++ (k,
                    Handler params metadata
                      (simplify_expr_pruning prune tenv_handler fit handler))
                   :: handlers)
               body')
          = true).
        { rewrite let_cont_recursive_sets.
          cbn [handler_sets_well_formed].
          rewrite Hhandler', Hothers. reflexivity. }
        split; [ exact Hall' | ].
        exact (let_cont_recursive_handler_typed_valid tenv before k params
                 metadata handler _ handlers body' tenv_handler Henv
                 (eqb_prop _ _ Hbinds) Hhandler Hhandler' Hall Hall'
                 Hhandler_refines). }
      destruct Hone as [Hall1 Hrefines1].
      specialize
        (IH
           (before ++ [(k, simplify_handler_pruning prune true tenv fit h)])).
      rewrite <- !app_assoc in IH. cbn [app] in IH.
      destruct (IH Hall1) as [Hall2 Hrefines2].
      split; [ exact Hall2 | ].
      exact (refines_valid_trans tenv _ _ _ Hrefines1 Hrefines2). }
    destruct (Hall [] Hsets1) as [Hsets2 Hrefines2].
    cbn [app] in Hsets2, Hrefines2.
    split; [ exact Hsets2 | ].
    exact (refines_valid_trans tenv _ _ _ Hfirst Hrefines2).
  - split; [ reflexivity | ].
    apply refines_under_valid, simplify_apply_refines_under.
  - split; [ reflexivity | ].
    apply refines_under_valid, expr_equivalent_under_refines.
    apply simplify_apply_cont_equivalent.
  - exact (simplify_switch_pruning_sound prune tenv fit scrutinee arms Hprune
             Hknown).
  - split; [ reflexivity | ].
    apply refines_under_valid, expr_refines_under_refl.
Qed.

Theorem simplify_expr_pruning_sets : simplify_expr_pruning_sets_statement.
Proof.
  intros prune tenv fit e Hprune Hknown Hsets.
  exact (proj1 (simplify_expr_pruning_sound prune Hprune e tenv fit Hknown
                  Hsets)).
Qed.

Theorem simplify_expr_pruning_refines
    : simplify_expr_pruning_refines_statement.
Proof.
  intros prune tenv fit e Hprune Hknown vars conts statics hp rd Hvalid Hsat.
  assert (Hsets : sets_well_formed e = true) by exact (proj1 (proj1 Hvalid)).
  exact (proj2 (simplify_expr_pruning_sound prune Hprune e tenv fit Hknown
                  Hsets)
           vars conts statics hp rd_
             (run_valid_rest e vars conts statics _ rd_ Hvalid)
           Hsat).
Qed.

Theorem simplify_by_types_pruning_refines
    : simplify_by_types_pruning_refines_statement.
Proof.
  intros prune e Hprune Hsets. apply (from_valid_program e _ Hsets).
  apply simplify_expr_pruning_refines; [ exact Hprune | ].
  intros _. exact (conj create_invariant create_aliases_kinded).
Qed.

(* With a [prune] that answers nothing, by the same recursion. *)
Fixpoint simplify_expr_no_pruning_eq (e : expr) {struct e} :
  ∀ (tenv : Typing_env.t) (fit : bool),
    simplify_expr_pruning no_pruning tenv fit e = simplify_expr tenv e.
Proof.
  destruct e
    as [x declared d body | bindings body | k handler body
        | handlers body | ap | ac | scrutinee arms | ];
    intros tenv fit; cbn [simplify_expr_pruning simplify_expr].
  - cbv zeta.
    destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
      as [ | d' tenv' | ]; try reflexivity.
    + f_equal. apply simplify_expr_no_pruning_eq.
    + destruct (declare_variable tenv x declared) as [tenv' | ];
        [ | reflexivity ].
      f_equal. apply simplify_expr_no_pruning_eq.
  - cbv zeta. f_equal; [ | apply simplify_expr_no_pruning_eq ].
    generalize (add_static_bindings tenv bindings).
    generalize (fit && statics_keep_fit tenv bindings).
    intros fit' tenv'.
    induction bindings as [ | binding bindings IH]; [ reflexivity | ].
    cbn [map]. f_equal; [ | exact IH ].
    destruct binding as [id c | id | sym const]; try reflexivity.
    destruct c as [params my_closure ret exn metadata code_body].
    cbn [simplify_static_binding_pruning simplify_code_pruning
           simplify_static_binding simplify_code].
    destruct (code_env_in tenv' params my_closure metadata)
      as [tenv_code | ];
      [ | reflexivity ].
    cbv zeta. rewrite (simplify_expr_no_pruning_eq code_body tenv_code fit').
    reflexivity.
  - cbv zeta. rewrite (simplify_expr_no_pruning_eq body tenv fit).
    destruct handler as [params metadata handler]. f_equal.
    assert (Hdefault :
      simplify_handler_pruning no_pruning false tenv fit
        (Handler params metadata handler)
      = simplify_handler false tenv (Handler params metadata handler)).
    { cbn [simplify_handler_pruning simplify_handler].
      destruct (handler_env tenv params metadata) as [tenv_handler | ];
        [ | reflexivity ].
      cbv zeta.
      rewrite (simplify_expr_no_pruning_eq handler tenv_handler fit).
      reflexivity. }
    destruct (find_one_use tenv k (Handler params metadata handler)
                (simplify_expr tenv body))
      as [[[target xs] ks] | ]; [ | exact Hdefault ].
    rewrite (simplify_expr_no_pruning_eq handler target), Hdefault.
    reflexivity.
  - f_equal; [ | apply simplify_expr_no_pruning_eq ].
    induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
    cbn [map]. f_equal; [ | exact IH ].
    destruct h as [params metadata handler].
    cbn [simplify_handler_pruning simplify_handler].
    destruct (handler_env tenv params metadata) as [tenv_handler | ];
      [ | reflexivity ].
    cbv zeta.
    rewrite (simplify_expr_no_pruning_eq handler tenv_handler fit).
    reflexivity.
  - reflexivity.
  - reflexivity.
  - unfold simplify_switch_pruning, no_pruning.
    destruct (fit && types_fit_b tenv
              && Typing_env.mem_simple (Some Name_mode.normal) tenv
                   scrutinee);
      reflexivity.
  - reflexivity.
Qed.

Theorem simplify_expr_no_pruning : simplify_expr_no_pruning_statement.
Proof. intros tenv fit e. apply simplify_expr_no_pruning_eq. Qed.

(* ======================================================================== *)
(* With the meet that the contract is proved of                             *)
(* ======================================================================== *)

Theorem simplify_by_types_pruning_with_refines
    : simplify_by_types_pruning_with_refines_statement.
Proof.
  intros join fuel e Hjoin Hsets.
  apply simplify_by_types_pruning_refines; [ | exact Hsets ].
  apply simplify_switch_rewrite_sound.
  exact (FlambdaSimplifySwitchProofs.meet_to_prove_bottom_sound join fuel
           Hjoin).
Qed.

Theorem prune_by_types_refines : prune_by_types_refines_statement.
Proof.
  intros fuel e. unfold prune_by_types.
  destruct (sets_well_formed e) eqn:Hsets; [ | apply refines_refl ].
  apply simplify_by_types_pruning_with_refines; [ | exact Hsets ].
  exact FlambdaMeetExamplesProofs.forgetful_join_sound.
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_pruning_no_arm : example_pruning_no_arm_statement.
Proof. intros k0 k1. split; vm_compute; reflexivity. Qed.

Theorem example_pruning_comparison : example_pruning_comparison_statement.
Proof. intros k0 k1 k2. split; vm_compute; reflexivity. Qed.
