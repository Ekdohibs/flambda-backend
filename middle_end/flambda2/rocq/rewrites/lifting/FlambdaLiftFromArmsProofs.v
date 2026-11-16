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

(* Proofs for FlambdaLiftFromArmsSpec.v. Nothing here needs review.

   The second run first binds the group at the head of the unit. From
   then on the two runs are those of FlambdaLiftFromCodeProofs.v, with
   no code named: the second expression is the first with its [Let]s
   made aliases, and the symbols of those [Let]s are bound to what the
   [Let]s make (FlambdaLiftFromCodeProofs.eval_lifted_out_bindings,
   which is of the bindings put once each).
   So the first step of the first run is answered by two of the second,
   and every other step by the step that
   FlambdaLiftFromCodeProofs.lifted_out_step gives.

   The two runs may start from a continuation environment of what may
   follow a unit (FlambdaObserversSpec.observers), the empty one for
   the statements of FlambdaLiftFromArmsSpec.v, which have the test of
   the exits ([unit_lifted_plain]); without it, the values of two exits
   that answer each other are lifted ([unit_lifted_observed], for
   FlambdaLiftObservedSpec.v). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaHeapSpec FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaUnusedStaticsSpec
  FlambdaImportsSpec FlambdaObserversSpec FlambdaUnitFromProofs
  FlambdaHeapFreeSpec FlambdaHeapFreeProofs
  FlambdaLiftSpec FlambdaLiftProofs FlambdaLiftFromCodeSpec
  FlambdaLiftFromCodeProofs FlambdaLiftFromArmsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The run of the unit lifted                                               *)
(* ======================================================================== *)

Section Unit.

Variable u : unit_interface.
Variable lets : list (path * lift_to).
Variable e : expr.

(* The static environment and the heap that the two runs start from:
   the environment holds no code and binds no name that the unit lifted
   binds. And the continuation environment that they start from: what
   may follow a unit (FlambdaObserversSpec.observers); the empty one is
   such. *)
Variable statics0 : static_env.
Variable hp0 : heap.
Variable rd0 : read_record.
Variable conts0 : cont_env.

(* The unit lifted, where something is lifted. *)
Let bindings : list static_binding :=
  once_each [] (lifted_out_bindings lets root e).
Let lifted : expr := Let_static bindings (lift_from_code [] lets root e).

Hypothesis Hwf : well_formed_unit u lifted = true.
Hypothesis Hsome : bindings <> [].
Hypothesis Hno_code : static_code statics0 = [].
Hypothesis Hfree : static_names_free_in statics0 lifted = true.
Hypothesis Hobservers : observers [] conts0.

(* The second state is one that the run of the unit lifted reaches. *)
Definition reached_lifted (s' : state) : Prop :=
  ∃ n : nat, run n (State lifted [] conts0 statics0 hp0 rd0) = Step s'.

(* The two states are the initial ones; or they are related as in
   FlambdaLiftFromCodeProofs.v and the second is reached. *)
Definition unit_lifted (s s' : state) : Prop :=
  (s = State e [] conts0 statics0 hp0 rd0
   ∧ s' = State lifted [] conts0 statics0 hp0 rd0)
  ∨ (state_lifted_out [] s s' ∧ reached_lifted s').

(* The first step of the second run binds the group; the states are
   then related. *)
Lemma head_bound :
  ∃ statics' : static_env,
    step (State lifted [] conts0 statics0 hp0 rd0)
    = Step (State (lift_from_code [] lets root e) [] conts0 statics' hp0 rd0)
    ∧ state_lifted_out [] (State e [] conts0 statics0 hp0 rd0)
        (State (lift_from_code [] lets root e) [] conts0 statics' hp0 rd0).
Proof.
  pose proof
    (well_formed_never_faults_under u [] lifted conts0 statics0 hp0 rd0 0 _ Hwf
       Hno_code Hfree Hobservers eq_refl)
    as Hfault.
  unfold static_fault in Hfault. simpl in Hfault.
  apply orb_false_iff in Hfault as [_ Hfault].
  destruct (no_rebinding statics0 _ Hfault) as [Hdistinct Hfresh].
  destruct (eval_lifted_out_bindings [] lets root e statics0 Hdistinct Hfresh)
    as (statics' & Heval & Hcode & Hbound).
  pose proof (eval_static_bindings_extend [] _ statics0 statics' Heval)
    as Hextend.
  exists statics'. split.
  - unfold lifted. rewrite step_let_static. fold bindings in Heval.
    rewrite Heval. reflexivity.
  - apply State_lifted_out;
      [ intros x v [=]
      | exact (observers_lifted_out [] statics' [] conts0 Hobservers) |
      | exact Hbound | apply heap_lifted_refl ].
    split; [ intros id; rewrite Hcode, Hno_code; reflexivity | ].
    split.
    + intros s allocated Hs. exists allocated.
      split; [ exact (proj2 Hextend s allocated Hs) | ].
      apply allocated_lifted_refl.
    + intros id c Hlookup. rewrite Hno_code in Hlookup.
      discriminate Hlookup.
Qed.

(* A unit of which something is lifted begins with a [Let] or a
   [Let_cont]: its first step is not an exit. *)
Lemma first_step_no_exit (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) :
  step (State e [] conts0 statics0 hp0 rd0) <> Exit k vs hp rd.
Proof.
  unfold bindings in Hsome.
  destruct e
    as [ x xkind d body | group body | k0 h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in Hsome; try (exfalso; apply Hsome; reflexivity).
  - rewrite step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) xkind); discriminate.
  - rewrite step_let_static.
    destruct (eval_static_bindings [] group statics0); discriminate.
  - rewrite step_let_cont_non_recursive. discriminate.
  - rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); discriminate.
Qed.

(* A step of the first state is answered by the second: an exit by an
   exit with values lifted, a step by one or more steps. *)
Lemma unit_lifted_step (s s' : state) :
  unit_lifted s s' ->
  answered_heaps unit_lifted exit_lifted_out (step s) s'.
Proof.
  intros [[-> ->] | (Hlifted & [n Hn])].
  - destruct head_bound as (statics' & Hhead & Hrelated).
    assert (Hreached :
      reached_lifted
        (State (lift_from_code [] lets root e) [] conts0 statics' hp0 rd0)).
    { exists 1. simpl. rewrite Hhead. reflexivity. }
    destruct Hreached as [n Hn].
    pose proof (lifted_out_step [] _ _ Hrelated
      (well_formed_never_faults_under u [] _ conts0 statics0 hp0 rd0 n _ Hwf
         Hno_code Hfree Hobservers Hn)) as Hanswer.
    pose proof first_step_no_exit as Hno_exit.
    destruct (step (State e [] conts0 statics0 hp0 rd0)) as [t | k vs | ];
      unfold answered_heaps in *; [ | | exact I ].
    + destruct Hanswer as (m & t' & Hrun & Ht).
      exists (S m), t'. split.
      * change (S (S m)) with (1 + S m). rewrite run_add. simpl.
        rewrite Hhead. exact Hrun.
      * right. split; [ exact Ht | ].
        exists (n + S m). rewrite run_add, Hn. exact Hrun.
    + exfalso. exact (Hno_exit k vs hp rd_ eq_refl).
  - pose proof (lifted_out_step [] s s' Hlifted
      (well_formed_never_faults_under u [] _ conts0 statics0 hp0 rd0 n s' Hwf
         Hno_code Hfree Hobservers Hn)) as Hanswer.
    destruct (step s) as [t | k vs | ]; unfold answered_heaps in *;
      [ | exact Hanswer | exact I ].
    destruct Hanswer as (m & t' & Hrun & Ht).
    exists m, t'. split; [ exact Hrun | ].
    right. split; [ exact Ht | ].
    exists (n + S m). rewrite run_add, Hn. exact Hrun.
Qed.

(* The unit lifted, run from what may follow a unit, comes to what the
   unit comes to, up to what an exit is compared by. *)
Lemma unit_lifted_observed :
  observed_refines (Step (State e [] conts0 statics0 hp0 rd0))
    (Step (State lifted [] conts0 statics0 hp0 rd0)).
Proof.
  apply (stuttering_heaps_observed unit_lifted exit_lifted_out);
    [ intros last vs hp vs' hp' [Hvs Hheap];
      exact (conj (lifted_same_scalars _ vs vs' Hvs)
               (heap_alike_keys _ hp hp' Hheap))
    | exact unit_lifted_step | left; split; reflexivity ].
Qed.

End Unit.

(* With the test of the exits, from no continuation. *)
Section Unit_refines.

Variable u : unit_interface.
Variable lets : list (path * lift_to).
Variable e : expr.
Variable statics0 : static_env.
Variable hp0 : heap.
Variable rd0 : read_record.

Let bindings : list static_binding :=
  once_each [] (lifted_out_bindings lets root e).
Let lifted : expr := Let_static bindings (lift_from_code [] lets root e).

Hypothesis Hwf : well_formed_unit u lifted = true.
Hypothesis Hplain : exits_plain_knowing [] [] e = true.
Hypothesis Hsome : bindings <> [].
Hypothesis Hno_code : static_code statics0 = [].
Hypothesis Hfree : static_names_free_in statics0 lifted = true.

(* The states are related, and the first is one of a unit that exits
   with symbols and constants only. *)
Definition unit_lifted_plain (s s' : state) : Prop :=
  unit_lifted lets e statics0 hp0 rd0 [] s s' ∧ plain_state s.

(* The values at an exit are lifted, and are constants and symbols: they
   are the same. *)
Lemma unit_lifted_plain_step (s s' : state) :
  unit_lifted_plain s s' ->
  answered_heaps unit_lifted_plain exit_plain (step s) s'.
Proof.
  intros [Hlifted Hplain_s].
  pose proof
    (unit_lifted_step u lets e statics0 hp0 rd0 [] Hwf Hsome Hno_code Hfree
       (Forall_nil _) s s' Hlifted)
    as Hanswer.
  pose proof (plain_step s Hplain_s) as Hstep.
  destruct (step s) as [t | k vs | ]; unfold answered_heaps in *;
    [ | | exact I ].
  - destruct Hanswer as (m & t' & Hrun & Ht).
    exists m, t'. split; [ exact Hrun | split; assumption ].
  - destruct Hanswer as (vs' & hp' & rd' & Hexit & Hvs & Hheap).
    exists vs', hp', rd'. split; [ exact Hexit | ].
    split; [ exact (plain_values_lifted _ vs vs' Hstep Hvs) | exact Hheap ].
Qed.

(* The unit lifted refines the unit, if the unit exits with the empty
   heap only. *)
Lemma unit_lifted_refines :
  (∀ (k : continuation) (vs : list value) (hp : heap),
    behaves (State e [] [] statics0 hp0 rd0) (Exits k vs hp) ->
    hp = empty_heap) ->
  refines (State e [] [] statics0 hp0 rd0)
    (State lifted [] [] statics0 hp0 rd0).
Proof.
  intros Hempty.
  apply (stuttering_heaps_refines unit_lifted_plain exit_plain
           unit_lifted_plain_step).
  - intros k vs hp last vs' hp' Hexits [Hsame Hheap].
    split; [ exact Hsame | ].
    rewrite (Hempty k vs hp Hexits) in Hheap |- *.
    exact (heap_alike_empty _ hp' Hheap).
  - split;
      [ left; split; reflexivity
      | exact (plain_from e statics0 hp0 rd0 Hplain Hno_code) ].
Qed.

End Unit_refines.

Theorem lift_to_head_unit_program_refines :
  lift_to_head_unit_program_refines_statement.
Proof.
  intros u lets e e'. unfold lift_to_head_unit. cbv zeta.
  destruct (exits_plain_knowing [] [] e) eqn:Hplain; [ | discriminate ].
  destruct (expr_free_of_heap e) eqn:Hheap_free; [ | discriminate ].
  destruct (well_formed_unit u (lift_to_head lets e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-]. unfold program_refines.
  destruct (once_each [] (lifted_out_bindings lets root e))
    as [ | binding rest] eqn:Hbindings.
  - unfold lift_to_head. cbv zeta. rewrite Hbindings. apply refines_refl.
  - assert (Hsome : once_each [] (lifted_out_bindings lets root e) <> [])
      by (rewrite Hbindings; discriminate).
    assert (Hshape :
      lift_to_head lets e
      = Let_static (once_each [] (lifted_out_bindings lets root e))
          (lift_from_code [] lets root e)).
    { unfold lift_to_head. cbv zeta. rewrite Hbindings. reflexivity. }
    rewrite Hshape in Hwf |- *.
    exact (unit_lifted_refines u lets e empty_static_env empty_heap no_reads Hwf
             Hplain Hsome eq_refl (free_in_empty _)
             (fun k vs hp =>
                free_of_heap_exits_empty e empty_static_env k vs hp
                  Hheap_free eq_refl)).
Qed.

Theorem lift_to_head_unit_refines_imports :
  lift_to_head_unit_refines_imports_statement.
Proof.
  intros imports u lets e e'. unfold lift_to_head_unit. cbv zeta.
  destruct (exits_plain_knowing [] [] e) eqn:Hplain; [ | discriminate ].
  destruct (expr_free_of_heap e) eqn:Hheap_free; [ | discriminate ].
  destruct (well_formed_unit u (lift_to_head lets e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-]. unfold unit_refines.
  destruct (once_each [] (lifted_out_bindings lets root e))
    as [ | binding rest] eqn:Hbindings.
  - unfold lift_to_head. cbv zeta. rewrite Hbindings. intros _.
    apply refines_refl.
  - assert (Hsome : once_each [] (lifted_out_bindings lets root e) <> [])
      by (rewrite Hbindings; discriminate).
    assert (Hshape :
      lift_to_head lets e
      = Let_static (once_each [] (lifted_out_bindings lets root e))
          (lift_from_code [] lets root e)).
    { unfold lift_to_head. cbv zeta. rewrite Hbindings. reflexivity. }
    rewrite Hshape in Hwf |- *. intros Hfree.
    exact
      (unit_lifted_refines u lets e (imports_statics imports) empty_heap
         no_reads Hwf Hplain Hsome eq_refl Hfree
         (fun k vs hp =>
            free_of_heap_exits_empty e (imports_statics imports) k vs hp
              Hheap_free eq_refl)).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem lift_to_head_equivariant : lift_to_head_equivariant_statement.
Proof.
  intros r lets e. unfold lift_to_head. cbv zeta.
  rewrite (lifted_out_bindings_equivariant r lets root e).
  destruct (once_each [] (lifted_out_bindings lets root e))
    as [ | binding rest] eqn:Hbindings; [ reflexivity | ].
  pose proof
    (once_each_closed r _ [] (lifted_out_bindings_closed r lets root e))
    as Hclosed.
  rewrite Hbindings in Hclosed. simpl in Hclosed |- *.
  rewrite Hclosed, (lift_from_code_equivariant r [] lets root e).
  reflexivity.
Qed.
