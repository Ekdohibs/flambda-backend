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

(* Proofs of the statements of FlambdaSimplifyCodeEnvSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextWhereSpec FlambdaContextWhereProofs
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaTypingEnvRefinementProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaProvers FlambdaProversSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPrimitivesProofs
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyProjectionsProofs
  FlambdaSimplifyApplySpec FlambdaSimplifyApplyProofs
  FlambdaSimplifyStaticsSpec FlambdaSimplifyStaticsProofs
  FlambdaSimplifyCodeEnvSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The environment of the body of code                                      *)
(* ======================================================================== *)

Lemma define_variables_checked_is
    (definitions : list (variable * flambda_kind)) :
  ∀ tenv tenv' : Typing_env.t,
    define_variables_checked tenv definitions = Some tenv' ->
    tenv' = Typing_env.add_definitions_of_params tenv definitions.
Proof.
  induction definitions as [ | [x kind] definitions IH]; intros tenv tenv';
    cbn [define_variables_checked].
  - intros [= <-]. reflexivity.
  - destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x)));
      [ | discriminate ].
    intros Hdefined. exact (IH _ tenv' Hdefined).
Qed.

Theorem code_env_in : code_env_in_statement.
Proof.
  intros tenv params my_closure metadata tenv_code.
  unfold FlambdaSimplifyCodeEnvSpec.code_env_in.
  destruct (Nat.eqb (length (unarize (cm_params_arity metadata)))
              (length params));
    [ | discriminate ].
  intros Hdefined.
  exact (define_variables_checked_is _ _ tenv_code Hdefined).
Qed.

(* The first test of [code_env_in]. *)
Lemma code_env_in_well_formed (tenv : Typing_env.t) (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    (tenv_code : Typing_env.t) :
  FlambdaSimplifyCodeEnvSpec.code_env_in tenv params my_closure metadata
  = Some tenv_code ->
  typing_env_well_formed (Typing_env.closure_env tenv) = true.
Proof.
  unfold FlambdaSimplifyCodeEnvSpec.code_env_in.
  destruct (Nat.eqb (length (unarize (cm_params_arity metadata)))
              (length params));
    [ | discriminate ].
  cbn [define_variables_checked].
  destruct (typing_env_well_formed (Typing_env.closure_env tenv));
    [ reflexivity | discriminate ].
Qed.

(* ======================================================================== *)
(* It is satisfied where the code runs                                      *)
(* ======================================================================== *)

(* No variable is in scope once the variables are out of scope. *)
Lemma closure_env_not_in_scope (tenv : Typing_env.t) (x : variable) :
  typing_env_well_formed (Typing_env.closure_env tenv) = true ->
  ~ in_scope_in_terms (Typing_env.closure_env tenv) x.
Proof.
  intros Hwf Hscope.
  apply in_scope_in_terms_lookup in Hscope.
  destruct Hscope as [ty [btm [Hlookup Hmode]]].
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [[_ [Hrange _]] _].
  destruct (Hrange x ty btm Hlookup) as [Hearliest Hnext].
  change (Typing_env.next_binding_time (Typing_env.closure_env tenv))
    with (Typing_env.next_binding_time tenv) in Hnext.
  change (Typing_env.min_binding_time (Typing_env.closure_env tenv))
    with (Typing_env.next_binding_time tenv) in Hmode.
  unfold Binding_time.With_name_mode.scoped_name_mode in Hmode.
  destruct (Binding_time.With_name_mode.binding_time btm
            <? Binding_time.earliest_var) eqn:Hbefore;
    [ apply Z.ltb_lt in Hbefore; lia | ].
  destruct (Typing_env.next_binding_time tenv
            <=? Binding_time.With_name_mode.binding_time btm) eqn:Hafter;
    [ apply Z.leb_le in Hafter; lia | ].
  simpl in Hmode. discriminate.
Qed.

Theorem closure_env_env_satisfies_strictly
    : closure_env_env_satisfies_strictly_statement.
Proof.
  intros tenv statics vars Hwf [_ [valuation [Hsatisfied _]]].
  split.
  - intros x Hscope. destruct (closure_env_not_in_scope tenv x Hwf Hscope).
  - exists valuation. split.
    + exact (proj2 (closure_env_satisfied _ statics valuation tenv)
               Hsatisfied).
    + intros x Hscope. destruct (closure_env_not_in_scope tenv x Hwf Hscope).
Qed.

(* The variables defined, bound as the machine binds parameters. *)
Lemma define_variables_checked_satisfied
    (definitions : list (variable * flambda_kind)) :
  ∀ (tenv tenv' : Typing_env.t) (statics : static_env) (vars vars' : env)
    (vs : list value),
    define_variables_checked tenv definitions = Some tenv' ->
    Forall2 (fun definition v => kind_of_value v = snd definition)
      definitions vs ->
    bind_params (map fst definitions) vs vars = Some vars' ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly tenv' statics vars'.
Proof.
  induction definitions as [ | [x kind] definitions IH];
    intros tenv tenv' statics vars vars' vs Hdefined Hkinds Hbind Hsat.
  - inversion Hkinds; subst. simpl in Hdefined, Hbind.
    injection Hdefined as <-. injection Hbind as <-. exact Hsat.
  - inversion Hkinds as [ | definition v definitions' vs' Hkind Hrest];
      subst.
    simpl in Hbind. cbn [define_variables_checked] in Hdefined.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_prop in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh.
    apply (IH _ tenv' statics ((x, v) :: vars) vars' vs' Hdefined Hrest
             Hbind).
    exact (add_variable_definition_env_satisfies_strictly tenv statics vars
             x kind v Hwf Hfresh Hkind Hsat).
Qed.

(* The environment of the code is satisfied at a call, in a static
   environment that satisfies the environment outside with no
   variables. *)
Lemma code_env_in_satisfied_from_closure (tenv : Typing_env.t)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (tenv_code : Typing_env.t)
    (statics : static_env) (vs : list value) (callee : value)
    (callee_vars : env) :
  FlambdaSimplifyCodeEnvSpec.code_env_in tenv params my_closure metadata
  = Some tenv_code ->
  env_satisfies_strictly (Typing_env.closure_env tenv) statics [] ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  env_satisfies_strictly tenv_code statics callee_vars.
Proof.
  unfold FlambdaSimplifyCodeEnvSpec.code_env_in.
  intros Henv Hsat Hkinds Hcallee Hbind.
  destruct (Nat.eqb (length (unarize (cm_params_arity metadata)))
              (length params)) eqn:Hlength;
    [ | discriminate ].
  apply Nat.eqb_eq in Hlength.
  apply (define_variables_checked_satisfied
           ((my_closure, Value)
            :: combine params (unarize (cm_params_arity metadata)))
           (Typing_env.closure_env tenv) tenv_code statics [] callee_vars
           (callee :: vs) Henv).
  - apply Forall2_cons; [ exact Hcallee | ].
    rewrite <- Hkinds. apply kinds_of_values.
    exact (bind_params_length params vs _ callee_vars Hbind).
  - cbn [map fst]. rewrite (map_fst_combine params _ Hlength). exact Hbind.
  - exact Hsat.
Qed.

Theorem code_env_in_satisfied : code_env_in_satisfied_statement.
Proof.
  intros tenv params my_closure metadata tenv_code statics vars vs callee
    callee_vars Henv Hsat Hkinds Hcallee Hbind.
  apply (code_env_in_satisfied_from_closure tenv params my_closure metadata
           tenv_code statics vs callee callee_vars Henv);
    [ | exact Hkinds | exact Hcallee | exact Hbind ].
  exact (closure_env_env_satisfies_strictly tenv statics vars
           (code_env_in_well_formed tenv params my_closure metadata tenv_code
              Henv)
           Hsat).
Qed.

(* ======================================================================== *)
(* The body of code that a [Let_static] binds                               *)
(* ======================================================================== *)

Lemma closure_symbols_of_code (before : list static_binding) (id : code_id)
    (c c' : code) (after : list static_binding) :
  closure_symbols_of (before ++ Bind_code id c :: after)
  = closure_symbols_of (before ++ Bind_code id c' :: after).
Proof.
  induction before as [ | [id0 c0 | id0 | s const] before IH];
    cbn [app closure_symbols_of]; [ reflexivity | exact IH | exact IH | ].
  destruct (is_closure_const const); [ f_equal; exact IH | exact IH ].
Qed.

Lemma add_block_like_bindings_code (before : list static_binding)
    (id : code_id) (c c' : code) (after : list static_binding) :
  ∀ tenv : Typing_env.t,
    add_block_like_bindings tenv (before ++ Bind_code id c :: after)
    = add_block_like_bindings tenv (before ++ Bind_code id c' :: after).
Proof.
  induction before as [ | [id0 c0 | id0 | s const] before IH]; intros tenv;
    cbn [app add_block_like_bindings];
    [ reflexivity | apply IH | apply IH | apply IH ].
Qed.

Lemma add_closure_bindings_code (prior : Typing_env.t)
    (before : list static_binding) (id : code_id) (c c' : code)
    (after : list static_binding) :
  ∀ tenv : Typing_env.t,
    add_closure_bindings prior tenv (before ++ Bind_code id c :: after)
    = add_closure_bindings prior tenv (before ++ Bind_code id c' :: after).
Proof.
  induction before as [ | [id0 c0 | id0 | s const] before IH]; intros tenv;
    cbn [app add_closure_bindings];
    [ reflexivity | apply IH | apply IH | apply IH ].
Qed.

Theorem add_static_bindings_code : add_static_bindings_code_statement.
Proof.
  intros tenv before id c c' after. unfold add_static_bindings. cbv zeta.
  rewrite (closure_symbols_of_code before id c c' after),
    (add_block_like_bindings_code before id c c' after).
  apply add_closure_bindings_code.
Qed.

Theorem let_static_code_typed : let_static_code_typed_statement.
Proof.
  intros tenv before id params my_closure ret exn metadata code_body
    code_body' after body tenv_code Henv Hsource Htarget Hbody vars conts
    statics hp' rd' Hsat.
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
  (* The group, whatever its code, leaves a static environment that
     satisfies the environment outside the code. *)
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
  (* A state whose expression binds nothing static has the predicate
     only by its static environment. *)
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
  apply (FlambdaContextWhereProofs.let_static_code_refines_from inv before
           id params my_closure ret exn metadata code_body code_body' after
           body vars conts statics).
  - intros s s' Hinv Hstep. unfold inv in Hinv |- *.
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
  - unfold inv. right. exists source, body.
    split; [ reflexivity | ]. split.
    + unfold source. rewrite existsb_app. simpl. apply orb_true_r.
    + exact (Hgroup source eq_refl).
  - unfold inv. right. exists target, body.
    split; [ reflexivity | ]. split.
    + unfold target. rewrite existsb_app. simpl. apply orb_true_r.
    + apply Hgroup. apply add_static_bindings_code.
  - rewrite Hsource, Htarget. reflexivity.
  - intros vs callee callee_vars return_to exn_to statics0 hp rd Hbind Hkinds
      Hcallee.
    split.
    + split; intros Hinv; unfold inv; left;
        [ exact (Hcall _ _ _ _ _ rd_ Hsource Hinv)
        | exact (Hcall _ _ _ _ _ rd_ Htarget Hinv) ].
    + intros Hinv. apply Hbody.
      exact (code_env_in_satisfied_from_closure _ params my_closure metadata
               tenv_code statics0 vs callee callee_vars Henv
               (Hcall _ _ _ _ _ rd_ Hsource Hinv) Hkinds Hcallee Hbind).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma define_variables_checked_renamed (r : renaming)
    (definitions : list (variable * flambda_kind)) :
  ∀ tenv : Typing_env.t,
    define_variables_checked (rename_typing_env r tenv)
      (map
         (fun definition =>
            (rename_variable r (fst definition), snd definition))
         definitions)
    = option_map (rename_typing_env r)
        (define_variables_checked tenv definitions).
Proof.
  induction definitions as [ | [x kind] definitions IH]; intros tenv;
    cbn [map define_variables_checked fst snd]; [ reflexivity | ].
  rewrite (typing_env_well_formed_rename r tenv).
  change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
  rewrite (mem_equivariant r None tenv (Name_var x)).
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x)));
    [ | reflexivity ].
  rewrite (add_variable_definition_equivariant r tenv x kind
             Name_mode.normal).
  apply IH.
Qed.

Lemma combine_map_fst {A B C : Type} (f : A -> C) (l : list A) :
  ∀ l' : list B,
    combine (map f l) l'
    = map (fun pair => (f (fst pair), snd pair)) (combine l l').
Proof.
  induction l as [ | a l IH]; intros [ | b l']; simpl;
    try reflexivity.
  f_equal. apply IH.
Qed.

Theorem code_env_in_rename : code_env_in_rename_statement.
Proof.
  intros r tenv params my_closure metadata.
  unfold FlambdaSimplifyCodeEnvSpec.code_env_in.
  rewrite length_map.
  destruct (Nat.eqb (length (unarize (cm_params_arity metadata)))
              (length params));
    [ | reflexivity ].
  rewrite (closure_env_equivariant r tenv), combine_map_fst.
  exact (define_variables_checked_renamed r
           ((my_closure, Value)
            :: combine params (unarize (cm_params_arity metadata)))
           (Typing_env.closure_env tenv)).
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_load_in_code : example_load_in_code_statement.
Proof.
  intros ret exn body bindings. subst bindings. cbv beta.
  set (metadata := make_code_metadata [Singleton Value] [Singleton Value]).
  set (block :=
         Bind_symbol example_s
           (Static_const_block 0 [Const (Tagged_immediate 7)])).
  pose (tenv_code :=
          Typing_env.add_definitions_of_params
            (Typing_env.add_variable_definition
               (Typing_env.closure_env
                  (add_static_bindings Typing_env.create [block]))
               example_my_closure Value Name_mode.normal)
            [(example_x, Value)]).
  apply (let_static_code_typed Typing_env.create [block] example_f
           [example_x] example_my_closure ret exn metadata _ _ [] body
           tenv_code);
    [ vm_compute; reflexivity | reflexivity | reflexivity | ].
  subst tenv_code.
  eapply expr_refines_under_trans.
  - (* The load is the [Simple] that the prover answers. *)
    match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_block_load tenv example_q Value
                    (Block_access_values (Some 0) (Some 1) Any_value)
                    Immutable 0 (Symbol example_s)
                    (alias_type_of Value (Symbol example_s)) body) as Hload
    end.
    cbv zeta in Hload.
    match type of Hload with
    | context [Provers.meet_block_field_simple ?tenv ?mode ?kind ?ty ?i] =>
      let answer :=
        eval vm_compute in
          (Provers.meet_block_field_simple tenv mode kind ty i)
      in
      replace (Provers.meet_block_field_simple tenv mode kind ty i)
        with answer in Hload by (vm_compute; reflexivity)
    end.
    cbv beta iota in Hload. apply Hload; [ discriminate | ].
    argument_computed.
  - (* The variable of the load is made equal to it. *)
    eapply (let_simple_refines_under _ example_q Value
              (Const (Tagged_immediate 7)) _ _ Value);
      [ vm_compute; reflexivity | vm_compute; reflexivity
      | intros ? [=]
      | first [ reflexivity | eexists; vm_compute; reflexivity ]
      | vm_compute; reflexivity
      | apply expr_equivalent_under_refines; apply apply_cont_canonical;
        apply Forall2_cons; [ canonical_computed | apply Forall2_nil ] ].
Qed.

(* The call is what the function of FlambdaSimplifyApplySpec.v answers,
   computed. *)
Local Ltac call_computed :=
  match goal with
  | |- expr_refines_under ?tenv (Apply ?ap) _ =>
    pose proof (simplify_apply_refines_under tenv ap) as Hcall;
    match type of Hcall with
    | expr_refines_under _ _ ?answer =>
      let computed := eval vm_compute in answer in
      replace answer with computed in Hcall by (vm_compute; reflexivity)
    end;
    exact Hcall
  end.

Theorem example_call_of_static_closure
    : example_call_of_static_closure_statement.
Proof.
  intros k exn ret exn' bindings. subst bindings. cbv beta.
  set (metadata := make_code_metadata [Singleton Value] [Singleton Value]).
  set (closure :=
         Bind_symbol example_s (Static_const_set_of_closures example_f [])).
  eapply expr_refines_under_trans.
  - (* The call in the code. *)
    pose (tenv_code :=
            Typing_env.add_definitions_of_params
              (Typing_env.add_variable_definition
                 (Typing_env.closure_env
                    (add_static_bindings Typing_env.create [closure]))
                 example_my_closure Value Name_mode.normal)
              [(example_x, Value)]).
    apply (let_static_code_typed Typing_env.create [closure] example_f
             [example_x] example_my_closure ret exn' metadata _
             (example_call (Symbol example_s) (Direct example_f)
                (Var example_x) ret exn')
             [] _ tenv_code);
      [ vm_compute; reflexivity | reflexivity | reflexivity | ].
    subst tenv_code. unfold example_call. call_computed.
  - (* The call in the body. *)
    apply let_static_typed. unfold example_call. call_computed.
Qed.
