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

(* Proofs of the statements of FlambdaPartialApplicationSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Arith Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaApplyProofs FlambdaContextProofs
  FlambdaWellFormedSpec FlambdaWellKindedSpec FlambdaWellKindedProofs
  FlambdaPartialApplicationSpec.
Import ListNotations.

Theorem apply_partial : apply_partial_statement.
Proof.
  intros ap vars conts statics hp rd callee vs extra called slots c Hcallee
    Hargs
    Hextra Hclosure Hkind Hcode Htupled Hcount Hkinds Htest.
  rewrite step_apply, Hcallee, Hargs, Hextra. unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  rewrite (shape_of_closure_call_not_full statics callee _ vs _ _ called
             slots c Hclosure Hkind Hcode Htupled);
    [ | intros Heq; rewrite Heq in Hcount; exact (Nat.lt_irrefl _ Hcount) ].
  rewrite Hkinds, kinds_eqb_refl, Htest. cbn [andb negb].
  assert (Hnot_over :
    Nat.ltb (length (code_params_arity c)) (length (ap_args_arity ap))
    = false)
    by (apply Nat.ltb_ge, Nat.lt_le_incl; exact Hcount).
  rewrite Hnot_over. reflexivity.
Qed.

Theorem apply_partial_application : apply_partial_application_statement.
Proof.
  intros ap vars conts statics hp rd closure applied vs extra applied_arity
    Hcallee Hargs Hextra Hkind Hnonempty Hkinds Hclosure.
  rewrite step_apply, Hcallee, Hargs, Hextra, Hkind. unfold call_function.
  rewrite (shape_of_call_closure statics closure Indirect_unknown_arity _ _ _
    Hclosure).
  unfold shape_of_call. cbn [partial_application_of].
  rewrite Hkinds, kinds_eqb_refl.
  destruct (ap_args_arity ap) as [ | component rest];
    [ destruct (Hnonempty eq_refl) | reflexivity ].
Qed.

Theorem apply_partial_application_undefined :
  apply_partial_application_undefined_statement.
Proof.
  intros ap vars conts statics hp rd closure applied applied_arity Hcallee Hbad.
  rewrite step_apply, Hcallee.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function, shape_of_call.
  destruct (ap_call_kind ap) as [named | ] eqn:Hkind;
    cbn [partial_application_of].
  - reflexivity.
  - destruct Hbad as [Hbad | Harity]; [ destruct (Hbad eq_refl) | ].
    rewrite Harity. reflexivity.
Qed.

Theorem apply_partial_application_kinds :
  apply_partial_application_kinds_statement.
Proof.
  intros ap vars conts statics hp rd closure applied vs applied_arity Hcallee
    Hargs
    Hkinds.
  rewrite step_apply, Hcallee, Hargs.
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function, shape_of_call.
  destruct (ap_call_kind ap) as [named | ]; cbn [partial_application_of];
    [ reflexivity | ].
  revert Hkinds.
  destruct (ap_args_arity ap) as [ | component rest]; intros Hkinds;
    [ reflexivity | ].
  destruct (kinds_eqb (map kind_of_value vs) (unarize (component :: rest)))
    eqn:Htest;
    [ | reflexivity ].
  apply kinds_eqb_eq in Htest. destruct (Hkinds Htest).
Qed.

(* The step of a call whose callee, arguments and extra arguments have
   values. *)
Lemma step_apply_values (f : simple) (kind : call_kind) (args : list simple)
    (k exn : continuation) (extra : list (simple * flambda_kind))
    (args_arity return_arity : arity) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (callee : value)
      (vs extra_vs : list value) :
  eval_simple vars f = Some callee ->
  eval_simples vars args = Some vs ->
  eval_simples vars (map fst extra) = Some extra_vs ->
  step
    (State (Apply (Apply_expr f kind args k exn extra args_arity return_arity))
       vars conts statics hp rd)
  = call_function callee kind vs args_arity return_arity (resolve k conts)
      (with_extra_args (resolve exn conts) extra_vs) statics hp rd.
Proof.
  intros Hf Hargs Hextra. rewrite step_apply. cbn [ap_callee ap_args].
  change
    (exn_extra_simples
       (Apply_expr f kind args k exn extra args_arity return_arity))
    with (map fst extra).
  rewrite Hf, Hargs, Hextra. reflexivity.
Qed.

(* The step of a call one of whose arguments, or extra arguments, has no
   value. *)
Lemma step_apply_no_value (f : simple) (kind : call_kind)
    (args : list simple) (k exn : continuation)
    (extra : list (simple * flambda_kind)) (args_arity return_arity : arity)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  eval_simples vars args = None ∨ eval_simples vars (map fst extra) = None ->
  step
    (State (Apply (Apply_expr f kind args k exn extra args_arity return_arity))
       vars conts statics hp rd)
  = Undefined.
Proof.
  intros Hnone. rewrite step_apply. cbn [ap_callee ap_args].
  change
    (exn_extra_simples
       (Apply_expr f kind args k exn extra args_arity return_arity))
    with (map fst extra).
  destruct (eval_simple vars f) as [callee | ]; [ | reflexivity ].
  destruct Hnone as [-> | ->]; [ reflexivity | ].
  destruct (eval_simples vars args); reflexivity.
Qed.

Theorem partial_then_apply : partial_then_apply_statement.
Proof.
  intros f kind args args' k k' exn exn' extra extra' p metadata args_arity
    args_arity' partial_arity return_arity vars conts statics hp rd callee
      called
    slots c Hf Hclosure Hkind Hcode Htupled Hcount Hnonempty Hfresh
    Hfresh'.
  (* the continuation is bound *)
  eapply refines_trans;
    [ apply equivalent_refines; eapply step_equivalent;
      apply step_let_cont_non_recursive
    | ].
  (* the first call *)
  match goal with
  | |- refines ?source ?target =>
    pose proof (step_undefined_refines source target) as Hwrong
  end.
  destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
    [ | apply Hwrong, step_apply_no_value; left; exact Hargs ].
  destruct (eval_simples vars (map fst extra)) as [extra_vs | ] eqn:Hextra;
    [ | apply Hwrong, step_apply_no_value; right; exact Hextra ].
  assert (Hnot_full : length args_arity <> length (code_params_arity c))
    by (intros Heq; rewrite Heq in Hcount; exact (Nat.lt_irrefl _ Hcount)).
  destruct
    (kinds_eqb (map kind_of_value vs) (unarize args_arity)
     && partial_or_over_application args_arity partial_arity
          (code_params_arity c) (code_result_arity c))
    eqn:Htests.
  2: { apply Hwrong.
       rewrite (step_apply_values _ _ _ _ _ _ _ _ _ _ _ _ rd_ _ _ _ Hf Hargs
                  Hextra).
       unfold call_function.
       rewrite (shape_of_call_closure statics callee _ _ _ _
                  (closure_not_partial statics callee _ Hclosure)),
         (shape_of_closure_call_not_full statics callee _ vs _ _ called
            slots c Hclosure Hkind Hcode Htupled Hnot_full),
         Htests.
       reflexivity. }
  apply andb_true_iff in Htests as [Hkinds Htest].
  apply kinds_eqb_eq in Hkinds.
  (* the handler takes the partial application, if it takes one [Value] *)
  destruct (kinds_eqb [Value] (hm_params_kinds metadata)) eqn:Hmetadata.
  2: { apply Hwrong.
       rewrite
         (apply_partial
            (Apply_expr f kind args k exn extra args_arity partial_arity)
            vars _ statics _ rd_ callee vs extra_vs called slots c Hf Hargs
              Hextra
            Hclosure Hkind Hcode Htupled Hcount Hkinds Htest).
       unfold apply_continuation. simpl ap_continuation.
       rewrite resolve_head. cbn [apply_cont_value map kind_of_value].
       rewrite Hmetadata. reflexivity. }
  eapply refines_trans.
  { apply equivalent_refines. eapply step_equivalent.
    rewrite
      (apply_partial
         (Apply_expr f kind args k exn extra args_arity partial_arity) vars _
         statics _ rd_ callee vs extra_vs called slots c Hf Hargs Hextra
           Hclosure
         Hkind Hcode Htupled Hcount Hkinds Htest).
    unfold apply_continuation. simpl ap_continuation. rewrite resolve_head.
    cbn [apply_cont_value map kind_of_value]. rewrite Hmetadata.
    reflexivity. }
  (* the second call: undefined unless its arguments and its extra
     arguments have values, and its arguments are of the kinds of its
     arity *)
  match goal with
  | |- refines ?source ?target =>
    pose proof (step_undefined_refines source target) as Hwrong'
  end.
  assert (Hp :
    eval_simple
      ((p, Value_partial_application callee vs args_arity) :: vars) (Var p)
    = Some (Value_partial_application callee vs args_arity))
    by (simpl; rewrite Pos.eqb_refl; reflexivity).
  destruct (eval_simples vars args') as [vs' | ] eqn:Hargs'.
  2: { apply Hwrong', step_apply_no_value. left.
       rewrite (eval_simples_extend vars p _ args' Hfresh). exact Hargs'. }
  destruct (eval_simples vars (map fst extra')) as [extra_vs' | ]
    eqn:Hextra'.
  2: { apply Hwrong', step_apply_no_value. right.
       rewrite (eval_simples_extend vars p _ _ Hfresh'). exact Hextra'. }
  assert (Hargs_p :
    eval_simples
      ((p, Value_partial_application callee vs args_arity) :: vars) args'
    = Some vs')
    by (rewrite (eval_simples_extend vars p _ args' Hfresh); exact Hargs').
  assert (Hextra_p :
    eval_simples
      ((p, Value_partial_application callee vs args_arity) :: vars)
      (map fst extra')
    = Some extra_vs')
    by (rewrite (eval_simples_extend vars p _ _ Hfresh'); exact Hextra').
  destruct (kinds_eqb (map kind_of_value vs') (unarize args_arity'))
    eqn:Hkinds'.
  2: { apply Hwrong'.
       rewrite (step_apply_values _ _ _ _ _ _ _ _ _ _ _ _ rd_ _ _ _ Hp Hargs_p
                  Hextra_p).
       unfold call_function, shape_of_call. cbn [partial_application_of].
       destruct args_arity' as [ | component rest]; [ reflexivity | ].
       rewrite Hkinds'. reflexivity. }
  (* the second call, and the one call *)
  apply equivalent_refines, step_eq_equivalent.
  assert (Hall : eval_simples vars (args ++ args') = Some (vs ++ vs'))
    by (rewrite eval_simples_app, Hargs, Hargs'; reflexivity).
  rewrite
    (step_apply_values _ _ _ _ _ _ _ _ _ _ _ _ rd_ _ _ _ Hp Hargs_p Hextra_p),
    (step_apply_values _ _ _ _ _ _ _ _ _ _ _ _ rd_ _ _ _ Hf Hall Hextra').
  unfold call_function.
  rewrite (shape_of_call_closure statics callee kind _ _ _
             (closure_not_partial statics callee _ Hclosure)),
    (shape_of_closure_call_kind statics callee kind _ _ _ called slots
       Hclosure Hkind).
  unfold shape_of_call. cbn [partial_application_of].
  destruct args_arity' as [ | component rest];
    [ destruct (Hnonempty eq_refl) | ].
  rewrite Hkinds'. reflexivity.
Qed.

Theorem apply_over : apply_over_statement.
Proof.
  intros ap vars conts statics hp rd callee vs extra called slots c Hcallee
    Hargs
    Hextra Hclosure Hkind Hcode Htupled Hcount Hkinds Htest.
  rewrite step_apply, Hcallee, Hargs, Hextra. unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  rewrite (shape_of_closure_call_not_full statics callee _ vs _ _ called
             slots c Hclosure Hkind Hcode Htupled);
    [ | intros Heq; rewrite Heq in Hcount; exact (Nat.lt_irrefl _ Hcount) ].
  rewrite Hkinds, kinds_eqb_refl, Htest. cbn [andb negb].
  apply Nat.ltb_lt in Hcount. rewrite Hcount. reflexivity.
Qed.

Theorem over_application_applied : over_application_applied_statement.
Proof.
  intros remaining remaining_arity return_arity return_to exn_to vs statics hp
    rd.
  apply apply_cont_value_over_application.
Qed.

(* ======================================================================== *)
(* An over-application and the term that splits it                          *)
(* ======================================================================== *)

(* Simulations in which a step of the first state is answered by one or
   more steps of the second, and so is an exit.
   (FlambdaSimulationProofs.stuttering_refines asks that an exit of the
   first is an exit of the second at once; here the machine applies what
   is left of an over-application in the step in which the term has yet
   to enter a handler.) *)
Section Late.

Variable R : state -> state -> Prop.

Definition answered (result : step_result) (s' : state) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp rd => ∃ n : nat, run (S n) s' = Exit k vs hp rd
  | Step t => ∃ (n : nat) (t' : state), run (S n) s' = Step t' ∧ R t t'
  end.

Hypothesis R_step : ∀ s s' : state, R s s' -> answered (step s) s'.

Lemma late_exit (n : nat) :
  ∀ (s s' : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s s' -> run n s = Exit k vs hp rd -> ∃ m : nat, run m s'
      = Exit k vs hp rd.
Proof.
  induction n as [ | n IH]; intros s s' k vs hp rd Hrel; simpl;
    [ discriminate | ].
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k' vs0 | ]; unfold answered in Hanswer.
  - destruct Hanswer as (j & t' & Hrun & Hrel'). intros Hexit.
    destruct (IH t t' k vs hp rd_ Hrel' Hexit) as [m Hm].
    exists (S j + m). rewrite run_add, Hrun. exact Hm.
  - intros [= <- <- <- ->]. destruct Hanswer as [j Hrun]. exists (S j). exact
    Hrun.
  - discriminate.
Qed.

(* If the second run has ended, so has the first: each of its steps uses
   up at least one of the second's. *)
Lemma late_ends (m : nat) :
  ∀ (s s' : state) (b : behaviour),
    R s s' ->
    ended (run m s') b ->
    ∃ n : nat,
      run n s = Undefined
      ∨ ∃ (k : continuation) (vs : list value) (hp : heap) (rd : read_record),
        run n s
        = Exit k vs hp rd.
Proof.
  induction m as [m IH] using (well_founded_induction lt_wf).
  intros s s' b Hrel Hended.
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k vs hp' rd' | ] eqn:Hstep; unfold answered in
    Hanswer.
  - destruct Hanswer as (j & t' & Hrun & Hrel').
    destruct (Nat.lt_ge_cases m (S j)) as [Hshort | Hlong].
    + exfalso. replace (S j) with (m + (S j - m)) in Hrun by lia.
      rewrite (ended_stable m _ s' b Hended) in Hrun.
      rewrite Hrun in Hended. destruct b; contradiction.
    + replace m with (S j + (m - S j)) in Hended by lia.
      rewrite run_add, Hrun in Hended.
      assert (Hless : m - S j < m) by lia.
      destruct (IH (m - S j) Hless t t' b Hrel' Hended) as [n Hn].
      exists (S n). simpl. rewrite Hstep. exact Hn.
  - exists 1. right. exists k, vs, hp', rd'. simpl. rewrite Hstep. reflexivity.
  - exists 1. left. simpl. rewrite Hstep. reflexivity.
Qed.

Lemma late_refines (s s' : state) : R s s' -> refines s s'.
Proof.
  intros Hrel.
  assert (Hexits :
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp)).
  { intros k vs hp [n [rd Hn]].
    destruct (late_exit n s s' k vs _ rd_ Hrel Hn) as [m Hm].
    exists m, rd. exact Hm. }
  split; [ exact Hexits | ].
  intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
  destruct (late_ends m s s' b Hrel Hended)
    as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
  - right. exists n. exact Hwrong.
  - left.
    assert (Hs : behaves s (Exits k vs hp))
      by (exists n, rd; exact Hexit).
    rewrite (behaviour_unique s' b (Exits k vs hp) Hb (Hexits k vs hp Hs)).
    exact Hs.
Qed.

(* The same from two states that are not related, whose first steps
   are. *)
Lemma late_refines_step (s s' : state) : answered (step s) s' -> refines s s'.
Proof.
  destruct (step s) as [t | k vs | ] eqn:Hstep; unfold answered.
  - intros (n & t' & Hrun & Hrel).
    apply (refines_trans s t);
      [ apply equivalent_refines, step_equivalent, Hstep | ].
    apply (refines_trans t t'); [ apply late_refines, Hrel | ].
    apply equivalent_refines, (equivalent_of_run 0 (S n)). symmetry.
    exact Hrun.
  - intros [n Hrun]. apply equivalent_refines, (equivalent_of_run 1 (S n)).
    rewrite Hrun. simpl. rewrite Hstep. reflexivity.
  - intros _. apply step_undefined_refines, Hstep.
Qed.

End Late.

(* What is left of an over-application, in the machine, and the handler
   that the term has for it: the handler takes the value returned as its
   one parameter and calls it with the arguments that remain, which its
   environment gives. Continuations are related that are the same but
   for that, inside one another. (The warning silenced is about
   induction schemes, which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive cont_split : cont_value -> cont_value -> Prop :=
  | Split_same (c : cont_value) : cont_split c c
  | Split_closure (h : continuation_handler) (vars : env)
      (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    Forall2 (fun kc kc' => fst kc = fst kc' ∧ cont_split (snd kc) (snd kc'))
      conts conts' ->
    cont_split (Closure h vars conts recursive)
      (Closure h vars conts' recursive)
  | Split_extra (c c' : cont_value) (extra : list value) :
    cont_split c c' -> cont_split (Extra_args c extra) (Extra_args c' extra)
  | Split_over (remaining : list value) (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_split return_to return_to' ->
    cont_split exn_to exn_to' ->
    cont_split
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to')
  | Split_handler (remaining : list value)
      (remaining_arity return_arity : arity) (return_to exn_to : cont_value)
      (p : variable) (args : list simple) (k exn : continuation)
      (extra : list (simple * flambda_kind)) (extra_vs : list value)
      (vars : env) (conts : cont_env) :
    eval_simples vars args = Some remaining ->
    eval_simples vars (map fst extra) = Some extra_vs ->
    ~ var_free_in_simples p args ->
    ~ var_free_in_simples p (map fst extra) ->
    cont_split return_to (resolve k conts) ->
    cont_split exn_to (with_extra_args (resolve exn conts) extra_vs) ->
    cont_split
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Closure
         (Handler [p] (make_handler_metadata [Value])
            (Apply
               (Apply_expr (Var p) Indirect_unknown_arity args k exn extra
                  remaining_arity return_arity)))
         vars conts []).

Definition conts_split (conts conts' : cont_env) : Prop :=
  Forall2 (fun kc kc' => fst kc = fst kc' ∧ cont_split (snd kc) (snd kc'))
    conts conts'.

Inductive states_split : state -> state -> Prop :=
  | States_split (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    conts_split conts conts' ->
    states_split (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd).

Lemma conts_split_refl (conts : cont_env) : conts_split conts conts.
Proof.
  induction conts as [ | kc conts IH]; [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply Split_same ].
Qed.

Lemma cont_split_inv (c c' : cont_value) :
  cont_split c c' ->
  match c with
  | Closure h vars conts recursive =>
    ∃ conts' : cont_env,
      c' = Closure h vars conts' recursive ∧ conts_split conts conts'
  | Outer k => c' = Outer k
  | Extra_args c0 extra =>
    ∃ c0' : cont_value, c' = Extra_args c0' extra ∧ cont_split c0 c0'
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    (∃ return_to' exn_to' : cont_value,
      c'
      = Over_application remaining remaining_arity return_arity return_to'
          exn_to'
      ∧ cont_split return_to return_to' ∧ cont_split exn_to exn_to')
    ∨ ∃ (p : variable) (args : list simple) (k exn : continuation)
        (extra : list (simple * flambda_kind)) (extra_vs : list value)
        (vars : env) (conts : cont_env),
        c'
        = Closure
            (Handler [p] (make_handler_metadata [Value])
               (Apply
                  (Apply_expr (Var p) Indirect_unknown_arity args k exn extra
                     remaining_arity return_arity)))
            vars conts []
        ∧ eval_simples vars args = Some remaining
        ∧ eval_simples vars (map fst extra) = Some extra_vs
        ∧ ~ var_free_in_simples p args
        ∧ ~ var_free_in_simples p (map fst extra)
        ∧ cont_split return_to (resolve k conts)
        ∧ cont_split exn_to (with_extra_args (resolve exn conts) extra_vs)
  end.
Proof.
  intros Hrel. inversion Hrel; subst.
  - destruct c' as [h vars conts recursive | k | c0 extra | ].
    + exists conts. split; [ reflexivity | apply conts_split_refl ].
    + reflexivity.
    + exists c0. split; [ reflexivity | apply Split_same ].
    + left. eexists _, _. split; [ reflexivity | ].
      split; apply Split_same.
  - eexists. split; [ reflexivity | assumption ].
  - eexists. split; [ reflexivity | assumption ].
  - left. eexists _, _. split; [ reflexivity | ]. split; assumption.
  - right. eexists _, _, _, _, _, _, _, _. split; [ reflexivity | ].
    repeat (split; [ eassumption | ]). assumption.
Qed.

Lemma with_extra_args_split (c c' : cont_value) (extra : list value) :
  cont_split c c' ->
  cont_split (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Split_extra, Hc ].
Qed.

Lemma resolve_split (k : continuation) (conts conts' : cont_env) :
  conts_split conts conts' -> cont_split (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Split_same | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

Lemma rec_closures_split
    (handlers all : list (continuation * continuation_handler)) (vars : env)
    (conts conts' : cont_env) :
  conts_split conts conts' ->
  conts_split (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
    (map (fun '(k, h) => (k, Closure h vars conts' all)) handlers).
Proof.
  intros Hconts.
  induction handlers as [ | [k h] handlers IH]; simpl; [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply Split_closure, Hconts ].
Qed.

(* What a step of the machine comes to is carried by what a step of the
   term comes to, in some more steps. *)
Definition carried (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp rd => ∃ n : nat, run_on n result' = Exit k vs hp rd
  | Step t =>
    ∃ (n : nat) (t' : state), run_on n result' = Step t' ∧ states_split t t'
  end.

Lemma carried_later (result : step_result) (s' : state) :
  carried result (step s') -> carried result (Step s').
Proof.
  destruct result as [t | k vs | ]; simpl; [ | | auto ].
  - intros (n & t' & Hrun & Hrel). exists (S n), t'. rewrite run_succ.
    split; [ exact Hrun | exact Hrel ].
  - intros [n Hrun]. exists (S n). rewrite run_succ. exact Hrun.
Qed.

Lemma carried_answered (s s' : state) :
  carried (step s) (step s') -> answered states_split (step s) s'.
Proof.
  destruct (step s) as [t | k vs | ]; unfold carried, answered; [ | | auto ].
  - intros (n & t' & Hrun & Hrel). exists n, t'. rewrite run_succ.
    split; [ exact Hrun | exact Hrel ].
  - intros [n Hrun]. exists n. rewrite run_succ. exact Hrun.
Qed.

Lemma call_code_split (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_split return_to return_to' ->
  cont_split exn_to exn_to' ->
  carried
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hexn. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (negb _); [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ];
    [ | exact I ].
  eexists 0, _. split; [ reflexivity | ].
  apply States_split.
  repeat (apply Forall2_cons || apply Forall2_nil); split; auto.
Qed.

Lemma run_call_split (return_partial return_partial' : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_split return_to return_to' ->
  cont_split exn_to exn_to' ->
  (∀ partial : value,
    carried (return_partial partial) (return_partial' partial)) ->
  carried
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape return_arity return_to' exn_to' statics hp
       rd).
Proof.
  intros Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_split; assumption.
  - apply call_code_split; [ apply Split_over; assumption | assumption ].
  - apply Hpartial.
  - exact I.
Qed.

(* What a call calls is of kind [Value]. *)
Lemma shape_of_call_callee_kind (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  kind_of_value callee = Value.
Proof.
  intros Hshape. destruct callee as [k | | | | | | | | ]; try reflexivity.
  exfalso.
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [Hclosure | (closure & applied & applied_arity & _ & Hcallee & _)];
    [ | discriminate ].
  unfold shape_of_closure_call in Hclosure. simpl in Hclosure.
  discriminate.
Qed.

Lemma apply_cont_value_split (c : cont_value) :
  ∀ (c' : cont_value) (vs : list value) (statics : static_env) (hp : heap)
    (rd : read_record),
    cont_split c c' ->
    carried (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics hp rd).
Proof.
  induction c
    as [[params metadata body] def_vars def_conts recursive | exit
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros c' vs statics hp rd Hc; apply cont_split_inv in Hc.
  - destruct Hc as (def_conts' & -> & Hconts). simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | exact I ].
    eexists 0, _. split; [ reflexivity | ].
    apply States_split, Forall2_app; [ | exact Hconts ].
    apply rec_closures_split, Hconts.
  - subst c'. simpl. exists 0. reflexivity.
  - destruct Hc as (c0' & -> & Hc). exact
    (IH c0' (vs ++ extra) statics hp rd_ Hc).
  - destruct vs as [ | v [ | v' vs]]; try exact I.
    destruct Hc
      as [(return_to' & exn_to' & -> & Hreturn & Hexn)
         | (p & args & k & exn & extra & extra_vs & vars & conts & -> & Hargs
            & Hextra & Hfresh & Hfresh' & Hreturn & Hexn)].
    + cbn [apply_cont_value].
      apply run_call_split; [ exact Hreturn | exact Hexn | ].
      intros partial. apply IHreturn, Hreturn.
    + destruct
        (shape_of_call statics v Indirect_unknown_arity remaining
           remaining_arity return_arity)
        as [shape | ] eqn:Hshape.
      2: { cbn [apply_cont_value]. rewrite Hshape. exact I. }
      pose proof (shape_of_call_callee_kind _ _ _ _ _ _ _ Hshape) as Hv.
      clear Hshape.
      cbn [apply_cont_value map hm_params_kinds make_handler_metadata].
      rewrite Hv, kinds_eqb_refl.
      cbn [bind_params rec_closures map app].
      apply carried_later.
      rewrite
        (step_apply_values (Var p) Indirect_unknown_arity args k exn extra _ _
           ((p, v) :: vars) conts statics _ rd_ v remaining extra_vs);
        [ | simpl; rewrite Pos.eqb_refl; reflexivity
        | rewrite (eval_simples_extend vars p v args Hfresh); exact Hargs
        | rewrite (eval_simples_extend vars p v _ Hfresh'); exact Hextra ].
      unfold call_function.
      apply run_call_split; [ exact Hreturn | exact Hexn | ].
      intros partial. apply IHreturn, Hreturn.
Qed.

Lemma call_function_split (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_split return_to return_to' ->
  cont_split exn_to exn_to' ->
  carried
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hexn. unfold call_function.
  apply run_call_split; [ exact Hreturn | exact Hexn | ].
  intros partial. apply apply_cont_value_split, Hreturn.
Qed.

Lemma step_split (s s' : state) :
  states_split s s' -> carried (step s) (step s').
Proof.
  intros [e vars conts conts' statics hp rd Hconts].
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step.
  - destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    eexists 0, _. split; [ reflexivity | apply States_split, Hconts ].
  - destruct (eval_static_bindings vars bindings statics) as [statics' | ];
      [ | exact I ].
    eexists 0, _. split; [ reflexivity | apply States_split, Hconts ].
  - eexists 0, _. split; [ reflexivity | ].
    apply States_split, Forall2_cons; [ | exact Hconts ].
    split; [ reflexivity | apply Split_closure, Hconts ].
  - destruct (handlers_bind_statics handlers); [ exact I | ].
    eexists 0, _. split; [ reflexivity | ].
    apply States_split, Forall2_app; [ | exact Hconts ].
    apply rec_closures_split, Hconts.
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    apply call_function_split;
      [ apply resolve_split, Hconts
      | apply with_extra_args_split, resolve_split, Hconts ].
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. apply apply_cont_value_split, resolve_split.
    exact Hconts.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    eexists 0, _. split; [ reflexivity | apply States_split, Hconts ].
  - exact I.
Qed.

Lemma firstn_app_exact {A : Type} (l l' : list A) :
  firstn (length l) (l ++ l') = l.
Proof.
  rewrite firstn_app, firstn_all, Nat.sub_diag. simpl. apply app_nil_r.
Qed.

Lemma skipn_app_exact {A : Type} (l l' : list A) :
  skipn (length l) (l ++ l') = l'.
Proof. rewrite skipn_app, skipn_all, Nat.sub_diag. reflexivity. Qed.

(* A full call at an arity that passes the tests is the full call at
   the arity of the parameters of the code. *)
Lemma call_code_split_at_params (closure : value) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_split return_to return_to' ->
  cont_split exn_to exn_to' ->
  carried
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs (code_params_arity c) return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hexn. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (kinds_eqb (map kind_of_value vs) (unarize args_arity)) eqn:Hkinds;
    [ | exact I ].
  destruct
    (kinds_per_parameter_eqb (unarize_per_parameter args_arity)
       (unarize_per_parameter (code_params_arity c)))
    eqn:Hparams;
    [ | exact I ].
  apply kinds_eqb_eq in Hkinds. apply kinds_per_parameter_eqb_eq in Hparams.
  assert (Hsame :
    kinds_eqb (map kind_of_value vs) (unarize (code_params_arity c)) = true).
  { apply kinds_eqb_eq. rewrite Hkinds.
    apply unarize_per_parameter_eq, Hparams. }
  assert (Hrefl :
    kinds_per_parameter_eqb (unarize_per_parameter (code_params_arity c))
      (unarize_per_parameter (code_params_arity c))
    = true)
    by (apply kinds_per_parameter_eqb_eq; reflexivity).
  rewrite Hsame, Hrefl. cbn [andb].
  destruct (negb (arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ];
    [ | exact I ].
  eexists 0, _. split; [ reflexivity | ].
  apply States_split.
  repeat (apply Forall2_cons || apply Forall2_nil); split; auto.
Qed.

Theorem over_application_split : over_application_split_statement.
Proof.
  intros f kind args args' k k' exn' extra p args_arity args_arity'
    return_arity vars conts statics hp rd callee called slots c Hf Hclosure
      Hcode
    Htupled Hcount Hsize Hnonempty Hdistinct Hfresh Hfresh'.
  (* the term binds its continuation *)
  eapply refines_trans;
    [ | apply equivalent_refines, equivalent_sym; eapply step_equivalent;
        apply step_let_cont_non_recursive ].
  apply (late_refines_step states_split);
    [ intros s s' Hrel; apply carried_answered, step_split, Hrel | ].
  apply carried_answered.
  destruct (eval_simples vars args) as [vs | ] eqn:Hargs.
  2: { rewrite (step_apply_no_value f kind (args ++ args'));
         [ exact I | left; rewrite eval_simples_app, Hargs; reflexivity ]. }
  destruct (eval_simples vars args') as [vs' | ] eqn:Hargs'.
  2: { rewrite (step_apply_no_value f kind (args ++ args'));
         [ exact I | left; rewrite eval_simples_app, Hargs, Hargs' ];
         reflexivity. }
  destruct (eval_simples vars (map fst extra)) as [extra_vs | ] eqn:Hextra.
  2: { rewrite (step_apply_no_value f kind (args ++ args'));
         [ exact I | right; exact Hextra ]. }
  assert (Hall : eval_simples vars (args ++ args') = Some (vs ++ vs'))
    by (rewrite eval_simples_app, Hargs, Hargs'; reflexivity).
  rewrite
    (step_apply_values _ _ _ _ _ _ _ _ _ _ _ _ rd_ _ _ _ Hf Hall Hextra),
    (step_apply_values _ _ _ _ _ _ _ _ _ _ _ _ rd_ _ _ _ Hf Hargs Hextra).
  rewrite resolve_head,
    (resolve_other exn' k _ conts (fun Heq => Hdistinct (eq_sym Heq))).
  unfold call_function.
  rewrite !(shape_of_call_closure statics callee _ _ _ _
              (closure_not_partial statics callee _ Hclosure)),
    (shape_of_closure_call_full statics callee (Direct called) vs _ _ called
       slots c Hclosure (or_intror eq_refl) Hcode Htupled eq_refl).
  (* a direct call of other code is undefined *)
  assert (Hkind :
    (kind = Indirect_unknown_arity ∨ kind = Direct called)
    ∨ ∃ named : code_id, kind = Direct named ∧ called <> named).
  { destruct kind as [named | ]; [ | auto ].
    destruct (Pos.eq_dec called named) as [<- | Hother]; [ auto | ].
    right. exists named. auto. }
  destruct Hkind as [Hkind | (named & -> & Hother)].
  2: { unfold shape_of_closure_call. rewrite Hclosure.
       apply Pos.eqb_neq in Hother. rewrite Hother. exact I. }
  assert (Hlonger :
    length (code_params_arity c) < length (args_arity ++ args_arity')).
  { rewrite length_app, Hcount.
    destruct args_arity'; [ destruct (Hnonempty eq_refl) | simpl; lia ]. }
  rewrite (shape_of_closure_call_not_full statics callee kind _ _ _ called
             slots c Hclosure Hkind Hcode Htupled)
    by (intros Heq; rewrite Heq in Hlonger; exact (Nat.lt_irrefl _ Hlonger)).
  destruct (negb _); [ exact I | ].
  apply Nat.ltb_lt in Hlonger. rewrite Hlonger. cbn [run_call].
  (* the arguments and the arity are split where the term splits them *)
  unfold partially_apply.
  rewrite <- Hsize, <- (eval_simples_length vars args vs Hargs),
    firstn_app_exact, skipn_app_exact, <- Hcount, firstn_app_exact,
    skipn_app_exact.
  apply call_code_split_at_params; [ | apply Split_same ].
  apply (Split_handler _ _ _ _ _ _ _ _ _ _ extra_vs);
    [ exact Hargs' | exact Hextra | exact Hfresh | exact Hfresh'
    | apply Split_same | apply Split_same ].
Qed.

Theorem examples_run : examples_run_statement.
Proof.
  intros e [<- | [<- | []]]; exists 20; eexists; vm_compute; reflexivity.
Qed.

Theorem examples_well_formed : examples_well_formed_statement.
Proof.
  intros e [<- | [<- | []]]; split; vm_compute; reflexivity.
Qed.

Theorem malformed_partial_application :
  malformed_partial_application_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem over_raises : over_raises_statement.
Proof. split; exists 20; eexists; vm_compute; reflexivity. Qed.

Theorem apply_extra_undefined : apply_extra_undefined_statement.
Proof.
  intros ap vars conts statics hp rd Hextra. rewrite step_apply, Hextra.
  destruct (eval_simple vars (ap_callee ap)); [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)); reflexivity.
Qed.
