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


(* Proofs of the statements of FlambdaTypingEnvRefinementSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaHeapSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaContextSpec
  FlambdaContextProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Refinement relative to a typing environment                              *)
(* ======================================================================== *)

Theorem expr_refines_under_weaken : expr_refines_under_weaken_statement.
Proof.
  intros tenv source target Hrefines vars conts statics hp rd _. apply Hrefines.
Qed.

Theorem expr_refines_under_create : expr_refines_under_create_statement.
Proof.
  intros source target. split.
  - intros Hunder vars conts statics hp rd. apply Hunder, create_satisfied.
  - apply expr_refines_under_weaken.
Qed.

Theorem expr_equivalent_under_refines
    : expr_equivalent_under_refines_statement.
Proof.
  intros tenv e e' Hequivalent vars conts statics hp rd Hsat.
  apply equivalent_refines, Hequivalent, Hsat.
Qed.

Theorem expr_refines_under_trans : expr_refines_under_trans_statement.
Proof.
  intros tenv e e' e'' Hfirst Hsecond vars conts statics hp rd Hsat.
  exact (refines_trans _ _ _ (Hfirst vars conts statics hp rd_ Hsat)
           (Hsecond vars conts statics hp rd_ Hsat)).
Qed.

Theorem bottom_refines_under : bottom_refines_under_statement.
Proof.
  intros tenv e e' Hbottom vars conts statics hp rd Hsat. exfalso.
  apply (bottom_not_satisfied tenv statics vars Hbottom).
  apply env_satisfies_strictly_env_satisfies. exact Hsat.
Qed.

(* ======================================================================== *)
(* Canonical elements                                                       *)
(* ======================================================================== *)

Lemma canonical_in_terms_sound (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (s canonical : simple) :
  env_satisfies_strictly tenv statics vars ->
  canonical_in_terms tenv s canonical ->
  eval_simple vars canonical = eval_simple vars s.
Proof.
  intros Hsat [Hmem [ty Hcanonical]].
  apply (type_simple_in_term_canonical tenv statics vars s canonical ty);
    [ | exact Hmem | exact Hcanonical ].
  apply env_satisfies_strictly_env_satisfies. exact Hsat.
Qed.

Lemma canonical_in_terms_simples (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (args args' : list simple) :
  env_satisfies_strictly tenv statics vars ->
  Forall2 (canonical_in_terms tenv) args args' ->
  eval_simples vars args' = eval_simples vars args.
Proof.
  intros Hsat Hargs.
  induction Hargs as [ | s s' args args' Hs Hargs IH]; [ reflexivity | ].
  simpl. rewrite (canonical_in_terms_sound tenv statics vars s s' Hsat Hs), IH.
  reflexivity.
Qed.

Theorem apply_cont_canonical : apply_cont_canonical_statement.
Proof.
  intros tenv k args args' action Hargs vars conts statics hp rd Hsat.
  apply step_eq_equivalent. rewrite !step_apply_cont. simpl.
  rewrite (canonical_in_terms_simples tenv statics vars args args' Hsat
             Hargs).
  reflexivity.
Qed.

Theorem let_simple_canonical : let_simple_canonical_statement.
Proof.
  intros tenv x kind s canonical body Hcanonical vars conts statics hp rd Hsat.
  apply let_equivalent_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | | intros v _ _;
     apply equivalent_refl ].
  exact (canonical_in_terms_sound tenv statics vars s canonical Hsat
           Hcanonical).
Qed.

Theorem switch_known : switch_known_statement.
Proof.
  intros tenv scrutinee i arms ac Hknown Hfind vars conts statics hp rd Hsat.
  apply step_equivalent. rewrite step_switch.
  rewrite <- (canonical_in_terms_sound tenv statics vars scrutinee _ Hsat
                Hknown).
  simpl. rewrite Hfind. reflexivity.
Qed.

(* ======================================================================== *)
(* The environment follows the term                                         *)
(* ======================================================================== *)

(* The variables in scope once a variable is defined at the normal name
   mode. *)
Lemma add_variable_definition_in_scope (tenv : Typing_env.t) (x : variable)
    (kind : flambda_kind) (y : variable) :
  typing_env_well_formed tenv = true ->
  (in_scope_in_terms
     (Typing_env.add_variable_definition tenv x kind Name_mode.normal) y
   <-> y = x ∨ (y <> x ∧ in_scope_in_terms tenv y)).
Proof.
  intros Hwf.
  pose proof (proj1 (typing_env_well_formed_iff tenv) Hwf) as Hconditions.
  destruct Hconditions as [[[_ Hmin] _] _].
  rewrite !in_scope_in_terms_lookup, add_variable_definition_lookup.
  change (Typing_env.min_binding_time
            (Typing_env.add_variable_definition tenv x kind Name_mode.normal))
    with (Typing_env.min_binding_time tenv).
  destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_var y) (Name_var x))
    as [Heq | Hne].
  - injection Heq as ->. split; [ intros _; left; reflexivity | ].
    intros _. eexists. eexists. split; [ reflexivity | ].
    unfold Binding_time.With_name_mode.scoped_name_mode. simpl.
    destruct (Typing_env.min_binding_time tenv
              <=? Typing_env.next_binding_time tenv) eqn:Hle;
      [ rewrite orb_true_r; reflexivity | ].
    apply Z.leb_gt in Hle. lia.
  - split.
    + intros Hscope. right. split; [ congruence | exact Hscope ].
    + intros [-> | [_ Hscope]]; [ congruence | exact Hscope ].
Qed.

Theorem add_variable_definition_env_satisfies_strictly
    : add_variable_definition_env_satisfies_strictly_statement.
Proof.
  intros tenv statics vars x kind v Hwf Hfresh Hkind
    [Hbound [valuation [Hsatisfied Hagrees]]].
  split.
  - intros y Hy.
    apply (add_variable_definition_in_scope tenv x kind y Hwf) in Hy.
    destruct Hy as [-> | [Hne Hy]].
    + exists v. apply lookup_head.
    + destruct (Hbound y Hy) as [w Hw]. exists w.
      exact (eq_trans (lookup_other y x v vars Hne) Hw).
  - exists ((x, v) :: valuation). split.
    + apply add_variable_definition_satisfied_bind;
        [ exact Hsatisfied | | exact Hkind ].
      exact (well_formed_not_mentioned tenv x Hwf Hfresh).
    + intros y Hy.
      apply (add_variable_definition_in_scope tenv x kind y Hwf) in Hy.
      destruct Hy as [-> | [Hne Hy]].
      * exact (eq_trans (lookup_head x v valuation)
                 (eq_sym (lookup_head x v vars))).
      * exact (eq_trans (lookup_other y x v valuation Hne)
                 (eq_trans (Hagrees y Hy)
                    (eq_sym (lookup_other y x v vars Hne)))).
Qed.

Theorem add_definitions_of_params_env_satisfies_strictly
    : add_definitions_of_params_env_satisfies_strictly_statement.
Proof.
  intros tenv statics vars vars' params. revert tenv vars.
  induction params as [ | [x kind] params IH];
    intros tenv vars vs Hinvariant Hdistinct Hfresh Hkinds Hbind Hsat.
  - inversion Hkinds; subst. simpl in Hbind. injection Hbind as <-.
    exact Hsat.
  - inversion Hkinds as [ | param v params' vs' Hkind Hrest]; subst.
    simpl in Hbind.
    change (negb (among x (map fst params)) && distinct (map fst params)
            = true) in Hdistinct.
    apply andb_true_iff in Hdistinct. destruct Hdistinct as [Hx Hdistinct].
    apply negb_true_iff in Hx.
    assert (Hfresh_x : Typing_env.mem None tenv (Name_var x) = false).
    { apply Hfresh. left. reflexivity. }
    apply (IH (Typing_env.add_variable_definition tenv x kind
                 Name_mode.normal) ((x, v) :: vars) vs').
    + apply add_variable_definition_invariant; assumption.
    + exact Hdistinct.
    + intros y Hin. rewrite mem_add_variable_definition.
      * apply Hfresh. right. exact Hin.
      * intros ->. apply among_In in Hin. congruence.
    + exact Hrest.
    + exact Hbind.
    + apply add_variable_definition_env_satisfies_strictly;
        [ exact (proj1 Hinvariant) | exact Hfresh_x | exact Hkind
        | exact Hsat ].
Qed.

(* The variables in scope once a variable is given a type. *)
Lemma replace_equation_in_scope (tenv : Typing_env.t) (x : variable)
    (ty : flambda_type) (y : variable) :
  in_scope_in_terms (Typing_env.replace_equation tenv (Name_var x) ty) y
  <-> in_scope_in_terms tenv y.
Proof.
  rewrite !in_scope_in_terms_lookup.
  change (Typing_env.min_binding_time
            (Typing_env.replace_equation tenv (Name_var x) ty))
    with (Typing_env.min_binding_time tenv).
  destruct (Pos.eq_dec y x) as [-> | Hne].
  - rewrite (replace_equation_lookup tenv (Name_var x) ty).
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv)) as [[ty0 btm0] | ]; simpl.
    + split.
      * intros [ty1 [btm1 [[= <- <-] Hmode]]]. exists ty0, btm0.
        split; [ reflexivity | exact Hmode ].
      * intros [ty1 [btm1 [[= <- <-] Hmode]]]. exists ty, btm0.
        split; [ reflexivity | exact Hmode ].
    + reflexivity.
  - rewrite (replace_equation_lookup_other tenv (Name_var x) (Name_var y) ty);
      [ reflexivity | congruence ].
Qed.

Theorem replace_equation_env_satisfies_strictly
    : replace_equation_env_satisfies_strictly_statement.
Proof.
  intros tenv statics vars x ty Hx Hfree Htype
    [Hbound [valuation [Hsatisfied Hagrees]]].
  split.
  - intros y Hy. apply Hbound. apply (replace_equation_in_scope tenv x ty y).
    exact Hy.
  - exists valuation. split.
    + apply replace_equation_satisfied; [ exact Hsatisfied | ].
      intros v Hv. right. rewrite value_of_name_var in Hv.
      rewrite (Hagrees x Hx) in Hv.
      apply (type_satisfied_coincide statics vars valuation v ty);
        [ | exact (Htype v Hv) ].
      intros y Hin. symmetry. exact (Hagrees y (Hfree y Hin)).
    + intros y Hy. apply Hagrees.
      apply (replace_equation_in_scope tenv x ty y). exact Hy.
Qed.

(* The variables in scope once two [Simple]s are made equal. *)
Lemma add_alias_in_scope (tenv : Typing_env.t)
    (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result) (y : variable) :
  Typing_env.add_alias tenv canonical_element1 canonical_element2
  = Some (Ok result) ->
  (in_scope_in_terms (Typing_env.add_alias_result_t result) y
   <-> in_scope_in_terms tenv y).
Proof.
  intros Hadd.
  pose proof (add_alias_frame tenv canonical_element1 canonical_element2
                result Hadd) as Hframe.
  cbv zeta in Hframe.
  destruct Hframe as [_ [_ [_ [_ [Hnames [_ [Hmin _]]]]]]].
  rewrite !in_scope_in_terms_lookup, Hnames, Hmin. reflexivity.
Qed.

(* A valuation that agrees with the machine on the variables in scope
   gives a [Simple] whose variable is in scope the machine's value. *)
Lemma agrees_eval_simple (tenv : Typing_env.t) (valuation vars : env)
    (s : simple) :
  agrees_in_terms tenv valuation vars ->
  (∀ x : variable, s = Var x -> in_scope_in_terms tenv x) ->
  eval_simple valuation s = eval_simple vars s.
Proof.
  intros Hagrees Hscope. destruct s as [x | sym | c]; [ | reflexivity.. ].
  exact (Hagrees x (Hscope x eq_refl)).
Qed.

Theorem add_alias_env_satisfies_strictly
    : add_alias_env_satisfies_strictly_statement.
Proof.
  intros tenv statics vars c1 c2 result Hscope Hadd Heval
    [Hbound [valuation [Hsatisfied Hagrees]]].
  split.
  - intros y Hy. apply Hbound.
    apply (add_alias_in_scope tenv c1 c2 result y Hadd). exact Hy.
  - exists valuation. split.
    + apply (add_alias_satisfied (fun _ => False) statics valuation tenv c1
               c2 result Hsatisfied Hadd).
      rewrite (agrees_eval_simple tenv valuation vars c1 Hagrees),
        (agrees_eval_simple tenv valuation vars c2 Hagrees);
        [ exact Heval | | ];
        intros x ->; apply Hscope; [ right | left ]; reflexivity.
    + intros y Hy. apply Hagrees.
      apply (add_alias_in_scope tenv c1 c2 result y Hadd). exact Hy.
Qed.

Theorem simple_of_kind_value : simple_of_kind_value_statement.
Proof.
  intros tenv statics vars s kind v [Hbound [valuation [Hsatisfied Hagrees]]]
    Hscope Hkind Heval.
  assert (Hname :
    ∀ (n : name) (ty : flambda_type),
      Typing_env.find tenv n (Some kind) = Some ty ->
      value_of_name valuation n = Some v ->
      kind_of_value v = kind).
  { intros n ty Hfind Hvalue.
    rewrite <- (find_kind tenv n kind ty Hfind).
    apply (has_type_unless_kind
             (exempt_by_poison valuation tenv n) statics valuation v ty).
    exact (find_satisfied (exempt_by_poison valuation tenv) statics valuation
             tenv n (Some kind) ty v Hsatisfied Hfind Hvalue). }
  destruct s as [x | sym | c]; simpl in Hkind, Heval.
  - destruct Hkind as [ty Hfind]. apply (Hname (Name_var x) ty Hfind).
    rewrite value_of_name_var, (Hagrees x (Hscope x eq_refl)). exact Heval.
  - destruct Hkind as [ty Hfind].
    exact (Hname (Name_symbol sym) ty Hfind Heval).
  - injection Heval as <-. rewrite <- Hkind. symmetry.
    apply kind_type_for_const.
Qed.

(* ======================================================================== *)
(* Going under binders                                                      *)
(* ======================================================================== *)

Theorem let_refines_under : let_refines_under_statement.
Proof.
  intros tenv x declared d body body' kind Hwf Hfresh Hkind Hbody vars conts
    statics hp rd Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv _ ].
  apply Hbody. apply add_variable_definition_env_satisfies_strictly;
    [ exact Hwf | exact Hfresh
    | exact (Hkind statics hp hp' rd rd' vars v Hsat Hv) | exact Hsat ].
Qed.

Theorem let_declared_refines_under : let_declared_refines_under_statement.
Proof.
  intros tenv x declared d body body' Hwf Hfresh Hbody vars conts statics hp rd
    Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
  apply Hbody. apply add_variable_definition_env_satisfies_strictly;
    [ exact Hwf | exact Hfresh | exact Hkind | exact Hsat ].
Qed.

Theorem let_typed_refines_under : let_typed_refines_under_statement.
Proof.
  intros tenv x declared d body body' ty Hwf Hfresh Hfree Htype Hbody vars
    conts statics hp rd Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv _ ].
  pose proof (Htype statics hp hp' rd rd' vars v Hsat Hv) as Hty.
  apply Hbody. apply replace_equation_env_satisfies_strictly.
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

(* Where a defining expression that reads no cell of the heap
   (FlambdaHeapSpec.named_reads_heap) runs, its value is the one that
   [eval_named] gives it (one way only: one that has a value by
   [eval_named] may not run, where a location among its arguments is
   not bound): [let_typed_refines_under] with the premise on that
   value. *)
Lemma let_typed_refines_under_pure (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) (body body' : expr)
    (ty : flambda_type) :
  named_reads_heap defining_expr = false ->
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars defining_expr = Some v ->
    type_satisfied statics vars v ty) ->
  expr_refines_under
    (Typing_env.replace_equation
       (Typing_env.add_variable_definition tenv x
          (FlambdaTypeGrammar.kind ty) Name_mode.normal)
       (Name_var x) ty)
    body body' ->
  expr_refines_under tenv (Let x declared defining_expr body)
    (Let x declared defining_expr body').
Proof.
  intros Hpure Hwf Hfresh Hfree Htype.
  apply (let_typed_refines_under tenv x declared defining_expr body body' ty
           Hwf Hfresh Hfree).
  intros statics hp hp' rd rd' vars v Hsat Hexec.
  exact (Htype statics vars v Hsat
           (proj1 (exec_named_pure statics hp hp' rd rd' vars defining_expr v
                     Hpure Hexec))).
Qed.

Theorem let_simple_refines_under : let_simple_refines_under_statement.
Proof.
  intros tenv x declared s body body' kind result Hwf Hfresh Hscope Hkind
    Hadd Hbody vars conts statics hp rd Hsat.
  apply let_refines_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | reflexivity | intros
     v Hv _ ]. simpl in Hv.
  assert (Hother :
    eval_simple ((x, v) :: vars) s = eval_simple vars s).
  { destruct s as [y | sym | c]; [ | reflexivity.. ]. simpl.
    apply lookup_other.
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)). }
  apply Hbody.
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

Theorem let_cont_body_refines_under : let_cont_body_refines_under_statement.
Proof.
  intros tenv k h body body' Hbody vars conts statics hp rd Hsat.
  apply let_cont_non_recursive_refines_step, Hbody, Hsat.
Qed.

Theorem let_cont_recursive_body_refines_under
    : let_cont_recursive_body_refines_under_statement.
Proof.
  intros tenv handlers body body' Hbody vars conts statics hp rd Hsat.
  apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
  apply Hbody, Hsat.
Qed.

Lemma bind_params_length (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    bind_params params vs vars = Some vars' -> length vs = length params.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars' Hbind;
    simpl in Hbind; try discriminate; [ reflexivity | ].
  simpl. f_equal. exact (IH vs _ vars' Hbind).
Qed.

Lemma map_fst_combine {A B : Type} (l : list A) :
  ∀ l' : list B, length l' = length l -> map fst (combine l l') = l.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] Hlength; simpl in *;
    try discriminate; [ reflexivity | ].
  f_equal. apply IH. congruence.
Qed.

Lemma kinds_of_values (params : list variable) :
  ∀ vs : list value,
    length vs = length params ->
    Forall2 (fun param v => kind_of_value v = snd param)
      (combine params (map kind_of_value vs)) vs.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] Hlength;
    simpl in *; try discriminate; [ apply Forall2_nil | ].
  apply Forall2_cons; [ reflexivity | ]. apply IH. congruence.
Qed.

(* A handler that refines under the environment with its parameters
   defined, at every choice of kinds, refines where it runs. *)
Lemma handler_refines_under_where (tenv : Typing_env.t)
    (statics : static_env) (vars : env) (params : list variable)
    (params_kinds : list flambda_kind) (handler handler' : expr) :
  typing_env_invariant tenv ->
  distinct params = true ->
  (∀ x : variable,
    x ∈ params -> Typing_env.mem None tenv (Name_var x) = false) ->
  (∀ kinds : list flambda_kind,
    length kinds = length params ->
    expr_refines_under
      (Typing_env.add_definitions_of_params tenv (combine params kinds))
      handler handler') ->
  env_satisfies_strictly tenv statics vars ->
  handler_refines_where (statics_extend statics) params params_kinds vars
    handler handler'.
Proof.
  intros Hinvariant Hdistinct Hfresh Hhandler Hsat.
  intros vs handler_vars conts' statics' hp rd Hextend _ Hbind.
  pose proof (bind_params_length params vs vars handler_vars Hbind)
    as Hlength.
  assert (Hkinds : length (map kind_of_value vs) = length params).
  { rewrite length_map. exact Hlength. }
  apply (Hhandler (map kind_of_value vs) Hkinds).
  apply (add_definitions_of_params_env_satisfies_strictly tenv statics' vars
           handler_vars (combine params (map kind_of_value vs)) vs
           Hinvariant).
  - rewrite (map_fst_combine params _ Hkinds). exact Hdistinct.
  - rewrite (map_fst_combine params _ Hkinds). exact Hfresh.
  - exact (kinds_of_values params vs Hlength).
  - rewrite (map_fst_combine params _ Hkinds). exact Hbind.
  - exact (env_satisfies_strictly_statics_extend tenv statics statics' vars
             Hextend Hsat).
Qed.

Theorem let_cont_handler_refines_under
    : let_cont_handler_refines_under_statement.
Proof.
  intros tenv k params metadata handler handler' body Hinvariant Hdistinct
    Hfresh Hhandler vars conts statics hp rd Hsat.
  apply (let_cont_handler_refines_where (statics_extend statics)
           (fun s s' Hs Hs' => statics_extend_trans statics s s' Hs Hs'));
    [ apply statics_extend_refl | ].
  exact (handler_refines_under_where tenv statics vars params
           (hm_params_kinds metadata) handler handler' Hinvariant Hdistinct
           Hfresh Hhandler Hsat).
Qed.

Theorem let_cont_recursive_handler_refines_under
    : let_cont_recursive_handler_refines_under_statement.
Proof.
  intros tenv before k params metadata handler handler' after body
    Hinvariant Hdistinct Hfresh Hbinds Hhandler vars conts statics hp rd Hsat.
  pose proof (handler_refines_under_where tenv statics vars params
                (hm_params_kinds metadata) handler handler' Hinvariant
                Hdistinct Hfresh Hhandler Hsat)
    as Hwhere.
  apply (let_cont_recursive_handlers_refine_where (statics_extend statics)
           (fun s s' Hs Hs' => statics_extend_trans statics s s' Hs Hs'));
    [ apply statics_extend_refl
    | unfold handlers_bind_statics; rewrite !existsb_app; simpl;
      rewrite Hbinds; reflexivity
    | | ];
    (apply Forall2_app; [ apply handlers_related_refl | ]);
    (apply Forall2_cons; [ | apply handlers_related_refl ]);
    (split; [ reflexivity | ]); apply Handler_related; exact Hwhere.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* What the environment answers is computed. *)
Ltac canonical_computed :=
  split; [ vm_compute; reflexivity | eexists; vm_compute; reflexivity ].

Theorem example_constant_propagated : example_constant_propagated_statement.
Proof.
  intros k. apply expr_refines_under_create.
  eapply (let_simple_refines_under Typing_env.create example_p Value
            (Const (Tagged_immediate 7)) _ _ Value).
  - reflexivity.
  - reflexivity.
  - intros y [=].
  - reflexivity.
  - vm_compute. reflexivity.
  - eapply expr_refines_under_trans.
    + apply expr_equivalent_under_refines.
      apply (let_simple_canonical _ example_x Value (Var example_p)
               (Const (Tagged_immediate 7))).
      canonical_computed.
    + eapply (let_simple_refines_under _ example_x Value
                (Const (Tagged_immediate 7)) _ _ Value).
      * vm_compute. reflexivity.
      * vm_compute. reflexivity.
      * intros y [=].
      * reflexivity.
      * vm_compute. reflexivity.
      * apply expr_equivalent_under_refines, apply_cont_canonical.
        apply Forall2_cons; [ canonical_computed | apply Forall2_nil ].
Qed.

Theorem example_switch_known : example_switch_known_statement.
Proof.
  intros k0 k1. apply expr_refines_under_create.
  eapply (let_simple_refines_under Typing_env.create example_i
            (Naked_number Kind_naked_immediate) (Const (Naked_immediate 1))
            _ _ (Naked_number Kind_naked_immediate)).
  - reflexivity.
  - reflexivity.
  - intros y [=].
  - reflexivity.
  - vm_compute. reflexivity.
  - apply expr_equivalent_under_refines.
    apply (switch_known _ (Var example_i) 1); [ | reflexivity ].
    canonical_computed.
Qed.

Theorem example_loop : example_loop_statement.
Proof.
  intros k j exit. apply expr_refines_under_create.
  apply (let_cont_recursive_handler_refines_under Typing_env.create [] k
           [example_p] _ _ _ []).
  - apply create_invariant.
  - reflexivity.
  - intros x [<- | []]. reflexivity.
  - reflexivity.
  - intros [ | kind [ | ? ?]]; try discriminate. intros _.
    destruct kind as [ | [] | | ];
      (eapply (let_simple_refines_under _ example_x Value
                 (Const (Tagged_immediate 7)) _ _ Value);
       [ vm_compute; reflexivity | vm_compute; reflexivity | intros y [=]
       | reflexivity | vm_compute; reflexivity | ]);
      (apply let_cont_handler_refines_under;
       [ split;
         [ vm_compute; reflexivity
         | intros n ty Hlookup; vm_compute in Hlookup; discriminate ]
       | reflexivity | intros x [] | ]);
      intros [ | ? ?]; try discriminate; intros _;
      apply expr_equivalent_under_refines, apply_cont_canonical;
      (apply Forall2_cons; [ canonical_computed | apply Forall2_nil ]).
Qed.
