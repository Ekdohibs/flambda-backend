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


(* Proofs of FlambdaSimplifyCarriedSpec.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaUnusedContsSpec FlambdaNominalSpec FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaTypingEnvExamplesSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyApplySpec
  FlambdaSimplifyStaticsSpec FlambdaSimplifyCodeEnvSpec
  FlambdaSimplifyHandlersSpec FlambdaSimplifyPassSpec
  FlambdaSimplifyPassProofs FlambdaSimplifyPassRenamingSpec
  FlambdaSimplifyPassRenamingProofs FlambdaSimplifyPruningSpec
  FlambdaSimplifyPruningProofs FlambdaSimplifyCarriedSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The test, where the boolean does not answer it                           *)
(* ======================================================================== *)

Lemma well_formed_or_known_eq (fit : bool) (tenv : Typing_env.t) :
  known_to_fit tenv fit ->
  well_formed_or_known fit tenv = typing_env_well_formed tenv.
Proof.
  intros Hknown. unfold well_formed_or_known.
  destruct fit; [ | reflexivity ].
  cbn [orb]. symmetry. exact (proj1 (proj1 (Hknown eq_refl))).
Qed.

(* A variable that is defined keeps the two conditions. *)
Lemma known_to_fit_define (fit : bool) (tenv : Typing_env.t) (x : variable)
    (kind : flambda_kind) :
  known_to_fit tenv fit ->
  Typing_env.mem None tenv (Name_var x) = false ->
  known_to_fit
    (Typing_env.add_variable_definition tenv x kind Name_mode.normal) fit.
Proof.
  intros Hknown Hfresh Hfit.
  exact (add_variable_definition_carried tenv x kind (Hknown Hfit) Hfresh).
Qed.

Lemma known_to_fit_closure_env (fit : bool) (tenv : Typing_env.t) :
  known_to_fit tenv fit -> known_to_fit (Typing_env.closure_env tenv) fit.
Proof.
  intros Hknown Hfit. destruct (Hknown Hfit) as [Hinvariant Haliases]. split.
  - exact (closure_env_invariant tenv Hinvariant).
  - exact (closure_env_aliases_kinded tenv Haliases).
Qed.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

(* The step is that of [simplify_let], and the environment of a step
   has the boolean that is answered with it. *)
Lemma simplify_let_carried_spec (fit : bool) (tenv : Typing_env.t)
    (x : variable) (d : named) :
  known_to_fit tenv fit ->
  fst (simplify_let_carried fit tenv x d) = simplify_let tenv x d
  ∧ (∀ (d' : named) (tenv' : Typing_env.t),
       simplify_let tenv x d = Step_let d' tenv' ->
       known_to_fit tenv'
         (fit && snd (simplify_let_carried fit tenv x d))).
Proof.
  intros Hknown. split.
  - unfold simplify_let_carried, simplify_let.
    rewrite (well_formed_or_known_eq fit tenv Hknown).
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x)));
      reflexivity.
  - intros d' tenv' Hlet.
    assert (Hsnd :
      snd (simplify_let_carried fit tenv x d) = let_keeps_fit tenv x d).
    { unfold simplify_let_carried.
      rewrite (well_formed_or_known_eq fit tenv Hknown).
      unfold simplify_let in Hlet.
      destruct (typing_env_well_formed tenv
                && negb (Typing_env.mem None tenv (Name_var x)));
        [ reflexivity | discriminate ]. }
    rewrite Hsnd.
    exact (let_keeps_fit_sound tenv tenv' fit x d d' Hknown Hlet).
Qed.

Lemma declare_variable_carried_eq (fit : bool) (tenv : Typing_env.t)
    (x : variable) (declared : flambda_kind) :
  known_to_fit tenv fit ->
  declare_variable_carried fit tenv x declared
  = declare_variable tenv x declared.
Proof.
  intros Hknown. unfold declare_variable_carried, declare_variable.
  rewrite (well_formed_or_known_eq fit tenv Hknown). reflexivity.
Qed.

Lemma declare_variable_known_to_fit (fit : bool) (tenv tenv' : Typing_env.t)
    (x : variable) (declared : flambda_kind) :
  known_to_fit tenv fit ->
  declare_variable tenv x declared = Some tenv' ->
  known_to_fit tenv' fit.
Proof.
  intros Hknown. unfold declare_variable.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
    [ | discriminate ].
  intros [= <-]. apply andb_prop in Htests. destruct Htests as [_ Hfresh].
  apply negb_true_iff in Hfresh.
  exact (known_to_fit_define fit tenv x declared Hknown Hfresh).
Qed.

Lemma let_env_carried_spec (fit : bool) (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) :
  known_to_fit tenv fit ->
  match let_env_carried fit tenv x declared d with
  | Some (tenv', fit') =>
    let_env tenv x declared d = Some tenv' ∧ known_to_fit tenv' fit'
  | None => let_env tenv x declared d = None
  end.
Proof.
  intros Hknown. unfold let_env_carried, let_env.
  destruct (simplify_let_carried_spec fit tenv x
              (resolve_value_slot tenv declared d) Hknown)
    as [Hstep Hkeeps].
  destruct (simplify_let_carried fit tenv x
              (resolve_value_slot tenv declared d))
    as [step keeps].
  cbn [fst snd] in Hstep, Hkeeps. rewrite <- Hstep in Hkeeps |- *.
  destruct step as [ | d' tenv' | ].
  - reflexivity.
  - split; [ reflexivity | ]. exact (Hkeeps d' tenv' eq_refl).
  - rewrite (declare_variable_carried_eq fit tenv x declared Hknown).
    destruct (declare_variable tenv x declared) as [tenv' | ] eqn:Hdeclare;
      [ | reflexivity ].
    split; [ reflexivity | ].
    exact (declare_variable_known_to_fit fit tenv tenv' x declared Hknown
             Hdeclare).
Qed.

(* ======================================================================== *)
(* The environments of code and of a handler                                *)
(* ======================================================================== *)

Lemma define_variables_carried_eq (fit : bool)
    (definitions : list (variable * flambda_kind)) :
  ∀ tenv : Typing_env.t,
    known_to_fit tenv fit ->
    define_variables_carried fit tenv definitions
    = define_variables_checked tenv definitions.
Proof.
  induction definitions as [ | [x kind] definitions IH]; intros tenv Hknown;
    cbn [define_variables_carried define_variables_checked];
    [ reflexivity | ].
  rewrite (well_formed_or_known_eq fit tenv Hknown).
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
    [ | reflexivity ].
  apply andb_prop in Htests. destruct Htests as [_ Hfresh].
  apply negb_true_iff in Hfresh.
  exact (IH _ (known_to_fit_define fit tenv x kind Hknown Hfresh)).
Qed.

Lemma code_env_in_carried_eq (fit : bool) (tenv : Typing_env.t)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) :
  known_to_fit tenv fit ->
  code_env_in_carried fit tenv params my_closure metadata
  = code_env_in tenv params my_closure metadata.
Proof.
  intros Hknown. unfold code_env_in_carried, code_env_in. cbv zeta.
  destruct (Nat.eqb (length (unarize (cm_params_arity metadata)))
              (length params));
    [ | reflexivity ].
  exact (define_variables_carried_eq fit _ _
           (known_to_fit_closure_env fit tenv Hknown)).
Qed.

Lemma handler_env_carried_eq (fit : bool) (tenv : Typing_env.t)
    (params : list variable) (metadata : handler_metadata) :
  known_to_fit tenv fit ->
  handler_env_carried fit tenv params metadata
  = handler_env tenv params metadata.
Proof.
  intros Hknown. unfold handler_env_carried, handler_env. cbv zeta.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | reflexivity ].
  exact (define_variables_carried_eq fit _ tenv Hknown).
Qed.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

Lemma bind_params_to_args_carried_spec (aliased : simple -> bool)
    (bindings : list (variable * flambda_kind * simple)) :
  ∀ (fit : bool) (tenv : Typing_env.t),
    known_to_fit tenv fit ->
    match bind_params_to_args_carried fit aliased tenv bindings with
    | Some (target, fit') =>
      bind_params_to_args aliased tenv bindings = Some target
      ∧ known_to_fit target fit'
    | None => bind_params_to_args aliased tenv bindings = None
    end.
Proof.
  induction bindings as [ | [[x kind] canonical] bindings IH];
    intros fit tenv Hknown;
    cbn [bind_params_to_args_carried bind_params_to_args].
  - split; [ reflexivity | exact Hknown ].
  - rewrite (well_formed_or_known_eq fit tenv Hknown).
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | reflexivity ].
    apply andb_prop in Htests. destruct Htests as [_ Hfresh].
    apply negb_true_iff in Hfresh.
    pose proof (known_to_fit_define fit tenv x kind Hknown Hfresh)
      as Hdefined.
    cbv zeta.
    destruct (aliased canonical && simple_in_terms_b tenv canonical
              && simple_of_kind_b tenv canonical kind);
      [ | exact (IH fit _ Hdefined) ].
    destruct (Typing_env.add_alias
                (Typing_env.add_variable_definition tenv x kind
                   Name_mode.normal)
                (Var x) canonical)
      as [[ | result | ] | ] eqn:Hadd;
      try exact (IH fit _ Hdefined).
    apply IH. intros Hfit. apply andb_true_iff in Hfit.
    destruct Hfit as [Hfit Halias].
    exact (alias_keeps_fit_carried _ _ _ _ result (Hdefined Hfit) Halias
             Hadd).
Qed.

Lemma use_env_carried_spec (fit : bool) (scope tenv : Typing_env.t)
    (params : list variable) (metadata : handler_metadata)
    (args : list simple) :
  known_to_fit tenv fit ->
  match use_env_carried fit scope tenv params metadata args with
  | Some (target, fit') =>
    use_env scope tenv params metadata args = Some target
    ∧ known_to_fit target fit'
  | None => use_env scope tenv params metadata args = None
  end.
Proof.
  intros Hknown. unfold use_env_carried, use_env. cbv zeta.
  destruct (simplify_simples tenv args) as [answers | ]; [ | reflexivity ].
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params)
            && Nat.eqb (length answers) (length params));
    [ | reflexivity ].
  exact (bind_params_to_args_carried_spec _ _ fit tenv Hknown).
Qed.

(* The environment of the one application at the end of the body. *)
Lemma leaf_use_carried_spec (fit : bool) (scope tenv : Typing_env.t)
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (body : expr) (tenv_use : Typing_env.t)
    (args : list simple) :
  known_to_fit tenv fit ->
  leaf_uses tenv k body = Some [(tenv_use, args)] ->
  match
    use_env_carried (fit && leaf_uses_keep_fit tenv k body) scope tenv_use
      params metadata args
  with
  | Some (target, fit_target) =>
    use_env scope tenv_use params metadata args = Some target
    ∧ known_to_fit target fit_target
  | None => use_env scope tenv_use params metadata args = None
  end.
Proof.
  intros Hknown Hleaf. apply use_env_carried_spec.
  intros Hfit. apply andb_true_iff in Hfit. destruct Hfit as [Hfit Hkeeps].
  apply (leaf_uses_carried tenv k body _ tenv_use args (Hknown Hfit) Hleaf
           Hkeeps).
  left. reflexivity.
Qed.

Fixpoint find_use_carried_spec (body : expr) {struct body} :
  ∀ (fit : bool) (scope tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler),
    known_to_fit tenv fit ->
    match find_use_carried fit scope tenv k h body with
    | Some (target, fit_target, xs, ks) =>
      find_use scope tenv k h body = Some (target, xs, ks)
      ∧ known_to_fit target fit_target
    | None => find_use scope tenv k h body = None
    end.
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros fit scope tenv k h Hknown; cbn [find_use_carried find_use];
    try reflexivity.
  - destruct (var_free_in_handler_b x h); [ reflexivity | ].
    pose proof (let_env_carried_spec fit tenv x declared d Hknown) as Henv.
    destruct (let_env_carried fit tenv x declared d) as [[tenv' fit'] | ];
      [ | rewrite Henv; reflexivity ].
    destruct Henv as [Henv Hknown']. rewrite Henv.
    pose proof (find_use_carried_spec rest fit' scope tenv' k h Hknown')
      as IH.
    destruct (find_use_carried fit' scope tenv' k h rest)
      as [[[[target fit_target] xs] ks] | ];
      [ | rewrite IH; reflexivity ].
    destruct IH as [IH Htarget]. rewrite IH.
    split; [ reflexivity | exact Htarget ].
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k h')
              && negb (cont_free_in_handler_b k' h));
      [ | reflexivity ].
    pose proof (find_use_carried_spec rest fit scope tenv k h Hknown) as IH.
    destruct (find_use_carried fit scope tenv k h rest)
      as [[[[target fit_target] xs] ks] | ];
      [ | rewrite IH; reflexivity ].
    destruct IH as [IH Htarget]. rewrite IH.
    split; [ reflexivity | exact Htarget ].
  - destruct h as [params metadata handler].
    destruct (leaf_uses tenv k (Apply_cont ac))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try reflexivity.
    pose proof (leaf_use_carried_spec fit scope tenv k params metadata
                  (Apply_cont ac) tenv_use args Hknown Hleaf)
      as Huse.
    destruct (use_env_carried
                (fit && leaf_uses_keep_fit tenv k (Apply_cont ac)) scope
                tenv_use params metadata args)
      as [[target fit_target] | ];
      [ | rewrite Huse; reflexivity ].
    destruct Huse as [Huse Htarget]. rewrite Huse.
    split; [ reflexivity | exact Htarget ].
  - destruct h as [params metadata handler].
    destruct (leaf_uses tenv k (Switch scrutinee arms))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try reflexivity.
    pose proof (leaf_use_carried_spec fit scope tenv k params metadata
                  (Switch scrutinee arms) tenv_use args Hknown Hleaf)
      as Huse.
    destruct (use_env_carried
                (fit && leaf_uses_keep_fit tenv k (Switch scrutinee arms))
                scope tenv_use params metadata args)
      as [[target fit_target] | ];
      [ | rewrite Huse; reflexivity ].
    destruct Huse as [Huse Htarget]. rewrite Huse.
    split; [ reflexivity | exact Htarget ].
Qed.

(* The applications of the end of a body, without the boolean that
   [gather_uses_carried] gives each. *)
Lemma forget_booleans (b : bool)
    (uses : list (Typing_env.t * list simple)) :
  map
    (fun use : Typing_env.t * bool * list simple =>
       (fst (fst use), snd use))
    (map (fun use => (fst use, b, snd use)) uses)
  = uses.
Proof.
  induction uses as [ | [tenv_use args] uses IH]; [ reflexivity | ].
  cbn [map fst snd]. rewrite IH. reflexivity.
Qed.

(* The applications gathered, each with an environment that its
   boolean is right about. *)
Fixpoint gather_uses_carried_spec (body : expr) {struct body} :
  ∀ (fit : bool) (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler),
    known_to_fit tenv fit ->
    match gather_uses_carried fit tenv k h body with
    | Some (uses, xs, ks) =>
      gather_uses tenv k h body
      = Some (map (fun use => (fst (fst use), snd use)) uses, xs, ks)
      ∧ ∀ use : Typing_env.t * bool * list simple,
          use ∈ uses -> known_to_fit (fst (fst use)) (snd (fst use))
    | None => gather_uses tenv k h body = None
    end.
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros fit tenv k h Hknown; cbn [gather_uses_carried gather_uses];
    try reflexivity.
  - destruct (var_free_in_handler_b x h); [ reflexivity | ].
    pose proof (let_env_carried_spec fit tenv x declared d Hknown) as Henv.
    destruct (let_env_carried fit tenv x declared d) as [[tenv' fit'] | ];
      [ | rewrite Henv; reflexivity ].
    destruct Henv as [Henv Hknown']. rewrite Henv.
    pose proof (gather_uses_carried_spec rest fit' tenv' k h Hknown') as IH.
    destruct (gather_uses_carried fit' tenv' k h rest)
      as [[[uses xs] ks] | ];
      [ | rewrite IH; reflexivity ].
    destruct IH as [IH Huses]. rewrite IH.
    split; [ reflexivity | exact Huses ].
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k' h));
      [ | reflexivity ].
    pose proof (gather_uses_carried_spec rest fit tenv k h Hknown) as IH.
    destruct (gather_uses_carried fit tenv k h rest)
      as [[[uses xs] ks] | ];
      [ | rewrite IH; reflexivity ].
    destruct IH as [IH Huses]. rewrite IH.
    destruct (cont_free_in_handler_b k h');
      [ | split; [ reflexivity | exact Huses ] ].
    destruct h' as [params' metadata' body'].
    destruct (negb (cont_free_in_handler_b k h)
              && forallb (fun x => negb (var_free_in_handler_b x h)) params');
      [ | reflexivity ].
    rewrite (handler_env_carried_eq fit tenv params' metadata' Hknown).
    destruct (handler_env tenv params' metadata') as [tenv_handler | ]
      eqn:Henv; [ | reflexivity ].
    pose proof (gather_uses_carried_spec body' fit tenv_handler k h
                  (handler_env_known_to_fit tenv fit params' metadata'
                     tenv_handler Hknown Henv))
      as IHh.
    destruct (gather_uses_carried fit tenv_handler k h body')
      as [[[uses' xs'] ks'] | ];
      [ | rewrite IHh; reflexivity ].
    destruct IHh as [IHh Huses']. rewrite IHh, map_app.
    split; [ reflexivity | ].
    intros use Hin. apply in_app_or in Hin.
    destruct Hin as [Hin | Hin]; [ exact (Huses' use Hin) | ].
    exact (Huses use Hin).
  - destruct (leaf_uses tenv k (Apply_cont ac)) as [uses | ] eqn:Hleaf;
      [ | reflexivity ].
    rewrite forget_booleans.
    split; [ reflexivity | ].
    intros use Hin. apply in_map_iff in Hin.
    destruct Hin as [[tenv_use args] [<- Hin]]. cbn [fst snd].
    intros Hfit. apply andb_true_iff in Hfit.
    destruct Hfit as [Hfit Hkeeps].
    exact (leaf_uses_carried tenv k _ uses tenv_use args (Hknown Hfit)
             Hleaf Hkeeps Hin).
  - destruct (leaf_uses tenv k (Switch scrutinee arms)) as [uses | ]
      eqn:Hleaf; [ | reflexivity ].
    rewrite forget_booleans.
    split; [ reflexivity | ].
    intros use Hin. apply in_map_iff in Hin.
    destruct Hin as [[tenv_use args] [<- Hin]]. cbn [fst snd].
    intros Hfit. apply andb_true_iff in Hfit.
    destruct Hfit as [Hfit Hkeeps].
    exact (leaf_uses_carried tenv k _ uses tenv_use args (Hknown Hfit)
             Hleaf Hkeeps Hin).
Qed.

Lemma find_gathered_use_carried_spec (fit : bool) (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr) :
  known_to_fit tenv fit ->
  match find_gathered_use_carried fit tenv k h body with
  | Some (target, fit_target, xs, ks) =>
    find_gathered_use tenv k h body = Some (target, xs, ks)
    ∧ known_to_fit target fit_target
  | None => find_gathered_use tenv k h body = None
  end.
Proof.
  intros Hknown. destruct h as [params metadata handler].
  unfold find_gathered_use_carried, find_gathered_use.
  pose proof (gather_uses_carried_spec body fit tenv k
                (Handler params metadata handler) Hknown)
    as Hgather.
  destruct (gather_uses_carried fit tenv k (Handler params metadata handler)
              body)
    as [[[uses xs] ks] | ];
    [ | rewrite Hgather; reflexivity ].
  destruct Hgather as [Hgather Huses]. rewrite Hgather.
  destruct uses as [ | [[tenv_use fit_use] args] [ | second others]];
    cbn [map fst snd]; try reflexivity.
  pose proof (use_env_carried_spec fit_use tenv tenv_use params metadata args
                (Huses (tenv_use, fit_use, args) (or_introl eq_refl)))
    as Huse.
  destruct (use_env_carried fit_use tenv tenv_use params metadata args)
    as [[target fit_target] | ];
    [ | rewrite Huse; reflexivity ].
  destruct Huse as [Huse Htarget]. rewrite Huse.
  split; [ reflexivity | exact Htarget ].
Qed.

Lemma find_one_use_carried_spec (fit : bool) (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr) :
  known_to_fit tenv fit ->
  match find_one_use_carried fit tenv k h body with
  | Some (target, fit_target, xs, ks) =>
    find_one_use tenv k h body = Some (target, xs, ks)
    ∧ known_to_fit target fit_target
  | None => find_one_use tenv k h body = None
  end.
Proof.
  intros Hknown. unfold find_one_use_carried, find_one_use.
  pose proof (find_use_carried_spec body fit tenv tenv k h Hknown) as Hfind.
  destruct (find_use_carried fit tenv tenv k h body)
    as [[[[target fit_target] xs] ks] | ].
  - destruct Hfind as [Hfind Htarget]. rewrite Hfind.
    split; [ reflexivity | exact Htarget ].
  - rewrite Hfind.
    exact (find_gathered_use_carried_spec fit tenv k h body Hknown).
Qed.

(* ======================================================================== *)
(* A term                                                                   *)
(* ======================================================================== *)

(* By the recursion of the two functions. *)
Fixpoint simplify_expr_carried_eq_aux (e : expr) {struct e} :
  ∀ (fit : bool) (tenv : Typing_env.t),
    known_to_fit tenv fit ->
    simplify_expr_carried fit tenv e = simplify_expr tenv e.
Proof.
  destruct e
    as [x declared d body | bindings body | k handler body
        | handlers body | ap | ac | scrutinee arms | ];
    intros fit tenv Hknown; cbn [simplify_expr_carried simplify_expr].
  - destruct (simplify_let_carried_spec fit tenv x
                (resolve_value_slot tenv declared d) Hknown)
      as [Hstep Hkeeps].
    destruct (simplify_let_carried fit tenv x
                (resolve_value_slot tenv declared d))
      as [step keeps].
    cbn [fst snd] in Hstep, Hkeeps. rewrite <- Hstep in Hkeeps |- *.
    destruct step as [ | d' tenv' | ].
    + reflexivity.
    + f_equal. apply simplify_expr_carried_eq_aux.
      exact (Hkeeps d' tenv' eq_refl).
    + rewrite (declare_variable_carried_eq fit tenv x declared Hknown).
      destruct (declare_variable tenv x declared) as [tenv' | ] eqn:Hdeclare;
        [ | reflexivity ].
      f_equal. apply simplify_expr_carried_eq_aux.
      exact (declare_variable_known_to_fit fit tenv tenv' x declared Hknown
               Hdeclare).
  - cbv zeta.
    pose proof (statics_keep_fit_sound tenv fit bindings Hknown) as Hknown'.
    f_equal; [ | apply simplify_expr_carried_eq_aux; exact Hknown' ].
    revert Hknown'.
    generalize (add_static_bindings tenv bindings).
    generalize (fit && statics_keep_fit tenv bindings).
    intros fit' tenv' Hknown'.
    induction bindings as [ | binding bindings IH]; [ reflexivity | ].
    cbn [map]. f_equal; [ | exact IH ].
    destruct binding as [id c | id | sym const]; try reflexivity.
    destruct c as [params my_closure ret exn metadata code_body].
    cbn [simplify_static_binding_carried simplify_code_carried
           simplify_static_binding simplify_code].
    rewrite (code_env_in_carried_eq fit' tenv' params my_closure metadata
               Hknown').
    destruct (code_env_in tenv' params my_closure metadata)
      as [tenv_code | ] eqn:Hcode;
      [ | reflexivity ].
    cbv zeta.
    rewrite (simplify_expr_carried_eq_aux code_body fit' tenv_code
               (code_env_in_known_to_fit tenv' fit' params my_closure
                  metadata tenv_code Hknown' Hcode)).
    reflexivity.
  - cbv zeta. rewrite (simplify_expr_carried_eq_aux body fit tenv Hknown).
    destruct handler as [params metadata handler]. f_equal.
    assert (Hdefault :
      simplify_handler_carried false fit tenv
        (Handler params metadata handler)
      = simplify_handler false tenv (Handler params metadata handler)).
    { cbn [simplify_handler_carried simplify_handler].
      rewrite (handler_env_carried_eq fit tenv params metadata Hknown).
      destruct (handler_env tenv params metadata) as [tenv_handler | ]
        eqn:Henv;
        [ | reflexivity ].
      cbv zeta.
      rewrite (simplify_expr_carried_eq_aux handler fit tenv_handler
                 (handler_env_known_to_fit tenv fit params metadata
                    tenv_handler Hknown Henv)).
      reflexivity. }
    pose proof (find_one_use_carried_spec fit tenv k
                  (Handler params metadata handler)
                  (simplify_expr tenv body) Hknown)
      as Hfind.
    destruct (find_one_use_carried fit tenv k
                (Handler params metadata handler) (simplify_expr tenv body))
      as [[[[target fit_target] xs] ks] | ];
      [ | rewrite Hfind; exact Hdefault ].
    destruct Hfind as [Hfind Htarget]. rewrite Hfind.
    rewrite (simplify_expr_carried_eq_aux handler fit_target target Htarget),
      Hdefault.
    reflexivity.
  - f_equal; [ | apply simplify_expr_carried_eq_aux; exact Hknown ].
    induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
    cbn [map]. f_equal; [ | exact IH ].
    destruct h as [params metadata handler].
    cbn [simplify_handler_carried simplify_handler].
    rewrite (handler_env_carried_eq fit tenv params metadata Hknown).
    destruct (handler_env tenv params metadata) as [tenv_handler | ] eqn:Henv;
      [ | reflexivity ].
    cbv zeta.
    rewrite (simplify_expr_carried_eq_aux handler fit tenv_handler
               (handler_env_known_to_fit tenv fit params metadata
                  tenv_handler Hknown Henv)).
    reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
Qed.

Theorem simplify_expr_carried_eq : simplify_expr_carried_eq_statement.
Proof.
  intros fit tenv e. exact (simplify_expr_carried_eq_aux e fit tenv).
Qed.

Theorem simplify_by_types_carried_eq : simplify_by_types_carried_eq_statement.
Proof.
  intros e. apply simplify_expr_carried_eq_aux.
  intros _. exact (conj create_invariant create_aliases_kinded).
Qed.

(* ======================================================================== *)
(* What the validation uses of the function                                 *)
(* ======================================================================== *)

(* The two facts that FlambdaValidateProofs.v and
   FlambdaValidatorEquivarianceProofs.v use of
   [simplify_by_types], of the function that is equal to it. *)
Lemma simplify_by_types_carried_refines (e : expr) :
  expr_refines e (simplify_by_types_carried e).
Proof.
  rewrite simplify_by_types_carried_eq. apply simplify_by_types_refines.
Qed.

Lemma simplify_by_types_carried_equivariant :
  rewrite_equivariant simplify_by_types_carried.
Proof.
  intros r e. rewrite !simplify_by_types_carried_eq.
  apply simplify_by_types_equivariant.
Qed.
