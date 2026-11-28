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

(* The proof of FlambdaReaperUnboxingRefinesClosuresParamsSpec.v.

   The computed parts are checked by evaluation. The refinement is
   FlambdaReaperUnboxingRefinesClosuresSimProofs.rr_refines applied to the
   two rebuilt terms, related by hand ([example_rr]): the unboxed names
   are those of the rebuild, with the changed names [x41] (the
   [my_closure] of the code) and [x60] added with no field ([M1]); the
   code 54 has the [my_closure] of the representation of the set, and its
   parameter [x40] unboxed ([ci1]); the call gives it the variable of the
   field of [x11] ([args_frame]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesClosuresOverSpec
  FlambdaReaperUnboxingRefinesClosuresSimProofs
  FlambdaReaperUnboxingRefinesClosuresParamsSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesCallsSimProofs.
Import ListNotations.

Module CSP := FlambdaReaperUnboxingRefinesCallsSimProofs.

Import FlambdaReaperUnboxing.Examples ParamsExample.
Local Open Scope positive_scope.

(* The representation of the set of [x60]. *)
Definition r1 : closure_repr :=
  {| cr_fields :=
       [(Field_value_slot 80,
         Unboxed
           [(Field_block 0 Value, Not_unboxed 81);
            (Field_block 1 Value, Not_unboxed 82)])];
     cr_function_slots := [(70, 83)];
     cr_current := 70 |}.

Definition fields45 : unboxed_fields :=
  [(Field_block 0 Value, Not_unboxed 61);
   (Field_block 1 Value, Not_unboxed 62)].

Definition fields40 : unboxed_fields := [(Field_block 0 Value, Not_unboxed 63)].

Definition fields10 : unboxed_fields :=
  [(Field_block 0 Value, Not_unboxed 65);
   (Field_block 1 Value, Not_unboxed 66)].

Definition fields11 : unboxed_fields := [(Field_block 0 Value, Not_unboxed 67)].

Definition M1 : list (node * unboxed_fields) :=
  [(Node_variable 41, []); (Node_variable 60, []);
   (Node_variable 45, fields45); (Node_variable 40, fields40);
   (Node_variable 10, fields10); (Node_variable 11, fields11)].

Definition CH1 (x : variable) : option closure_repr :=
  if (x =? 41) || (x =? 60) then Some r1 else None.

Definition ci1 : code_info :=
  Code_info (Some r1) 1 false [CSP.Psh_unboxed fields40] arity1.

Definition CS1 (id : code_id) : option code_info :=
  if id =? 54 then Some ci1 else None.

Definition RCH1 (_ : bool) (_ : variable) (_ : code_id) : Prop := True.
Definition W1 (_ : variable) : Prop := False.
Definition REM1 (_ : code_id) : Prop := True.

Lemma M1_distinct : distinct (field_variables M1) = true.
Proof. vm_compute. reflexivity. Qed.

Lemma CH1_cases (x : variable) (r : closure_repr) :
  CH1 x = Some r -> (x = 41 ∨ x = 60) ∧ r = r1.
Proof.
  unfold CH1. destruct ((x =? 41) || (x =? 60)) eqn:E; [ | discriminate ].
  intros H. injection H as <-. split; [ | reflexivity ].
  apply orb_true_iff in E as [E | E]; apply Pos.eqb_eq in E;
    [ left | right ]; exact E.
Qed.

Lemma CH1_M (x : variable) (r : closure_repr) :
  CH1 x = Some r -> fields_of M1 x = Some [].
Proof.
  intros H. destruct (CH1_cases x r H) as [[-> | ->] _]; reflexivity.
Qed.

Lemma CH1_fresh (x : variable) (r : closure_repr) :
  CH1 x = Some r -> among x (field_variables M1) = false.
Proof.
  intros H. destruct (CH1_cases x r H) as [[-> | ->] _]; reflexivity.
Qed.

Ltac not_heap := let q := fresh "q" in let a := fresh "a" in
  let H := fresh "H" in intros q a H; discriminate H.

(* The two rebuilt terms related. *)
Lemma example_rr : rr M1 CH1 CS1 RCH1 W1 REM1 [] first_rebuilt rebuilt.
Proof.
  unfold first_rebuilt, rebuilt.
  apply rr_static.
  - apply rr_bindings_code with (ci := ci1); [ | | apply rr_bindings_nil ].
    + (* The frame of the code: its parameter unboxed. *)
      split; [ reflexivity | ].
      split.
      { unfold params_frame. cbn [ci1 ci_params plain_shapes forallb].
        split; [ intros kind [H | []]; discriminate H | ].
        split; [ intros p [<- | []]; reflexivity | ].
        split.
        { exists [(40, Value)], [(63, Value)].
          split; [ reflexivity | split; [ reflexivity | ] ].
          exact (CSP.params_rel2_unboxed M1 [] [] 40 fields40 [] []
                   eq_refl eq_refl (CSP.params_rel2_nil M1 [])). }
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ vm_compute; reflexivity | ].
        apply CSP.sks_follow2_nil. }
      do 3 (split; [ reflexivity | ]).
      do 4 (split; [ reflexivity | ]).
      reflexivity.
    + (* The body of the code. *)
      cbn [code_body code_dead ci1 ci_never].
      eapply (rr_proj_unboxed M1 CH1 CS1 RCH1 W1 REM1 [] 45 _ 80 (Var 41) 41
                r1 fields45 _ [(61, 81, Value); (62, 82, Value)]);
        [ reflexivity | reflexivity | reflexivity | reflexivity
        | left; reflexivity | reflexivity | left; reflexivity
        | reflexivity | ].
      eapply (rr_read M1 CH1 CS1 RCH1 W1 REM1 [] 46 Value _
                (Field_block 0 Value) (Var 45) 45 61 fields45);
        [ reflexivity | reflexivity | left; reflexivity | reflexivity
        | left; reflexivity | ].
      eapply (rr_read M1 CH1 CS1 RCH1 W1 REM1 [] 47 Value _
                (Field_block 1 Value) (Var 45) 45 62 fields45);
        [ reflexivity | reflexivity | left; reflexivity | reflexivity
        | right; left; reflexivity | ].
      apply rr_let; [ reflexivity | reflexivity | not_heap | ].
      eapply (rr_read M1 CH1 CS1 RCH1 W1 REM1 [] 50 Value _
                (Field_block 0 Value) (Var 40) 40 63 fields40);
        [ reflexivity | reflexivity | left; reflexivity | reflexivity
        | left; reflexivity | ].
      apply rr_let; [ reflexivity | reflexivity | not_heap | ].
      apply rr_apply_cont; [ reflexivity | intros [] ].
  - (* The body of the unit. *)
    eapply (rr_alloc M1 CH1 CS1 RCH1 W1 REM1 [] 10 _ fields10
              [[(65, Value, int 7)]; [(66, Value, int 8)]]);
      [ reflexivity | reflexivity | reflexivity | | not_heap | ].
    { constructor; [ | constructor; [ | constructor ] ].
      - exists Value, (int 7). split; [ reflexivity | ].
        split; [ exists 0%nat; repeat split | reflexivity ].
      - exists Value, (int 8). split; [ reflexivity | ].
        split; [ exists 1%nat; repeat split | reflexivity ]. }
    eapply (rr_alloc M1 CH1 CS1 RCH1 W1 REM1 [] 11 _ fields11
              [[(67, Value, int 32)]]);
      [ reflexivity | reflexivity | reflexivity | | not_heap | ].
    { constructor; [ | constructor ].
      exists Value, (int 32). split; [ reflexivity | ].
      split; [ exists 0%nat; repeat split | reflexivity ]. }
    eapply (rr_set M1 CH1 CS1 RCH1 W1 REM1 [] 60 r1 [(70, 54)]
              [(80, Var 10)]);
      [ reflexivity | reflexivity | | | ].
    { constructor; [ | constructor ].
      exists (Var 10). split; [ reflexivity | ].
      exists 10, fields10, [(81, 65, Value); (82, 66, Value)].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; reflexivity. }
    { vm_compute. constructor; [ intros [H | []]; discriminate H | ].
      constructor; [ intros [] | constructor ]. }
    apply (rr_call M1 CH1 CS1 RCH1 W1 REM1 [] (call [Var 11] arity1) 54 ci1
             [Var 67] arity1);
      [ reflexivity | reflexivity | | | reflexivity | reflexivity
      | intros [] | intros [] ].
    + exists 60. split; reflexivity.
    + unfold args_frame. cbn [ci1 ci_params plain_shapes forallb].
      split; [ | split; [ reflexivity | vm_compute; reflexivity ] ].
      exact (CSP.args_rel2_unboxed M1 [] [] fields40 fields11 11
               [(63, 67, Value)] [] [] eq_refl eq_refl
               (CSP.args_rel2_beyond M1 [] [] eq_refl)).
Qed.

Theorem repr_params_example : repr_params_example_statement.
Proof.
  split; [ intros nr reach rem; vm_compute; reflexivity | ].
  do 5 (split; [ vm_compute; reflexivity | ]).
  exact (rr_refines M1 M1_distinct CH1 CH1_M CH1_fresh CS1 RCH1 W1 REM1
           first_rebuilt rebuilt (fun _ _ _ => I) (fun _ => I) example_rr).
Qed.
