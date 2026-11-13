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

(* The proofs of FlambdaReaperRebuildSpec.v. Nothing here needs review.

   The proof of [rebuild_refines] is a simulation of the run of the unit
   by the run of the rebuilt unit, in which a step of the first is
   answered by one step of the second, or by none where the rebuild
   removed a [Let] or a [Let_cont] (the expression of the first then
   shrinks), or by anything where the first goes wrong in the next step
   ([shrinking_refines]). The relation ([related]) has:
   - the invariant of FlambdaReaperFlowProofs.v on the first run
     ([flow_invariant]): where the values of the first run may be;
   - the facts of the expression of the first run in the graph
     ([expr_edges]), at toplevel, under the nodes of the parameters of
     the continuations in scope, which the relation of the continuations
     says are those of the handlers bound (the invariant of the flow has
     the facts too, but allows the nodes of the continuations of code
     being run, which are not those of the handlers);
   - an environment of the rebuild ([rebuild_env]) under which the
     expression of the second is the rebuild of that of the first;
   - the variables of the first run that the rebuild keeps bound in the
     second to related values ([env_ok], [rel]): the same constants and
     symbols (a constant at a node with any source, but the poison of
     [Region] and [Rec_info], which is anywhere), a block with the same
     tag and as many fields, each field
     that is used related and the others poison, a boxed number or
     poison where its contents
     is not used, an array of elements related at nodes with any usage,
     and a closure of the same set with the value slots that the rebuild
     keeps, each used one related; and, if the runs may have any
     ([partials], which the runs of the statements may), a partial
     application whose closure is related and observed equal, and whose
     arguments are related at the parameters that have a use of the
     witnesses that their groups reach from what the closure may come
     from; what the last of these witnesses returns first may come to
     where the partial application is. [rel] says which allocation the
     value may come from, as [flows] does, so that the fields kept are
     known;
   - the continuations that the expression of the second names, related
     in the two runs ([conts_related], [cont_rel]): both ways out of the
     term through a continuation of the unit, or handlers, the second
     the rebuild of the first, with the mask of the parameters kept; in
     code, a handler that the code returns to, and in the second its
     rebuild or the wrapper of the call, or a handler that the code
     raises to, with the extra arguments of the call, and in the second
     its rebuild with those it keeps, or what is left of an
     over-application, the same in the two runs, its arguments related at
     the witnesses that come after those of the code ([way_out],
     [over_rel]);
   - the static environments related ([statics_rel]), as the
     environments of variables are, for the symbols that have a use;
   - the same heap, and what the unit asks of the expression of the first
     ([expr_ok]): its [Switch]es are on what has any usage, or in code on
     what the code id uses, it gives the continuations of the unit one
     value and those of code as many as counted, it meets the check
     ([checked]: [no_code_no_slots] if [exact], otherwise the static
     constants and the projections of [any_kind_arguments]), it
     does not name the continuation of the wrappers,
     its calls that return to handlers in scope return values of their
     kinds, and its calls that raise to handlers in scope raise to
     exception handlers, which keep the exception, and give extra
     arguments to no other continuation ([fresh_kinded], [raises_kept]);
   - the projections of a set that the rebuild has pending
     ([pending_ok]).
   At a node with any usage, what the second run has for a value is the
   same but for the value slots of its closures, and the same if [exact]
   ([sim], [rel_au_sim]): an exit is observed equal in the two runs, and
   equal if [exact] ([exits_related]). A primitive of the default case
   sees no value slot ([sim_respects], with
   FlambdaPrimitiveSpec.eval_prim_related).

   The facts of the graph that the proof reads are those of
   FlambdaReaperFlowProofs.expr_edges, and two more: the scrutinee of a
   [Switch] has any usage ([traverse_scrutinees]), so that the rebuild
   keeps it; and the two facts that keep the tuple of a call of tupled
   code alive ([tupled_kept], from [prepare_code]: [traverse_keeps],
   [traverse_unit_keeps]). The rebuild keeps every variable of kind
   [Region] or [Rec_info] ([is_var_used]), used or not; such a variable
   holds the poison of its kind, the only value of the kind
   ([region_value]), given by a simple, by the identity or by a call,
   the same in the two runs ([region_rel], [rel_region_argument]).

   A direct call is answered by the call of the rebuilt code
   ([direct_answered], [call_entered]), after the binding of the wrapper
   if the rebuild gives the call one: the code of the callee is bound in
   the second run, rebuilt, as its code id has a use (the call reads it);
   the arguments kept are those at the parameters of the code that have a
   use, and have a use themselves ([known_mask_used], [known_mask_kept]);
   the call kind of the rebuild is the code of the callee or indirect
   ([rewrite_call_kind_cases]); the parameters and [my_closure] are
   related in the two runs, and the continuations of the code are the
   ways out or the handler that the call returns to, at the nodes of the
   code ([way_out_transport], [read_transports]). A return to a handler
   through the wrapper is answered by two steps, the wrapper's and the
   handler's ([apply_cont_answered]). The exception continuation of the
   code is that of the call with its extra arguments: a way out, or an
   exception handler in scope, to which the second run gives the extra
   arguments at the parameters kept ([rebuild_exn_extra_args]); a raise
   of the code to it enters the handler with the exception followed by
   them, the exception at a parameter that the handler keeps
   ([filter_kept_hd]). That a handler raised to is an exception handler
   of a non-recursive [Let_cont], and that no call gives extra arguments
   to a way out, comes from [well_formed_unit]
   ([raises_kept_of_well_formed]). What the body of code is asked comes
   from the unit ([Hcode_checks]), and in it the continuations of the
   code are given as many values as the flow fragment counts
   ([exits_bound_of_counted]).

   An indirect call of [tupled_calls] is answered whatever its shape
   ([indirect_answered], [answered_engine]). The callee is a closure, or
   a partial application of a closure of code that is not tupled. The
   witnesses of unknown arity of the code are in a chain from the
   allocation of the closure ([witness_chain], [code_wchain]), and stand
   for those of the call, group by group ([wchain_cover], from
   FlambdaReaperFlowProofs.indirect_pending), after those that a partial
   application reached ([wchain_shift_end]). The mask of the arguments
   keeps those at the parameters that have a use of the witnesses that
   the sources of the callee lead to, group by group: those of the code
   are among them ([unknown_mask_used]), and each is a source of the
   witness of the call of its group, so what it keeps has a use
   ([unknown_mask_kept]). So the arguments are related at the witnesses
   that their groups reach ([args_rel]), and with those that a partial
   application keeps ([args_rel_app]), at those of the code
   ([closure_called]). A full call enters the code ([code_entered]): the
   last witness of the code gives the results and the exception, which
   the last of the call reads ([call_reads]). An over-application enters
   the code with what is left of it, related at the result of the code,
   with the arguments left related at the witnesses that come after
   ([args_rel_skipn], [wchain_from_return]). A partial application goes
   where the call returns, related there. Values given to what is left
   of an over-application make a call of unknown arity, which returns
   where the over-application does; so returns and calls are answered
   together, by induction on the continuation ([returned_cases],
   [called_of_returned]). The code of the closure is used even where the
   call is a partial application: the first witness of the call reads
   its code id.

   A closure of tupled code takes the fields of the first argument, the
   tuple, followed by the other arguments ([tupled_called]): a full call,
   or an over-application. Its code has one witness of unknown arity,
   which the first witness of the call stands for, whose one parameter is
   the tuple, the parameters of the code being read off its fields. The
   code is used, so the two facts that keep the tuple alive give that
   parameter a use, though no field of the tuple has one; so the mask
   keeps the tuple, related there. The second run has a block of the same
   tag and as many fields, those used related and the others poison, all
   values: the machine reads it as the tuple of the code, and the
   parameters of the code that have a use get related fields
   ([tuple_fields_rel]). The arguments after the tuple are at the
   witnesses that come after that of the code, from what it returns, as
   for an over-application of other code.

   The two statements are the two modes of one proof
   ([rebuild_refines_checked]): [rebuild_refines] with [exact], for
   which a closure has no value slot, and [rebuild_refines_observably]
   without.

   The second run is that of the rebuild with changes of calling
   convention (FlambdaReaperCallingConventions.rebuild_expr_cc). With a
   table that has every code of the unit it is [rebuild_expr] case for
   case ([rebuild_expr_cc_unchanged]): [rebuild_refines_checked] runs it
   so. Otherwise ([rebuild_cc_refines_checked]) the table is closed
   under the rules of [cannot_change_calling_convention]
   ([Hcc_closed]), the witnesses of known arity read off a node agree
   ([Hwidgets]), and code is returned to as
   FlambdaReaperCallingConventionsSpec.ret_calls_ok says ([Hcode_rets],
   [code_rets_of_unit]: the kinds by [results_kinded],
   [ret_calls_ok_of_results]; no raise by [well_formed_unit]).
   A call of unknown arity runs code whose witness of unknown arity is
   used, as [call_reads] says of the first witness of the call
   ([call_reads_hu]), so code that keeps its calling convention
   ([unchanged_of_unknown]). A direct call of code that changes is known
   to call it ([direct_call_recognized]): the known witness of the code
   is used, by the call, and so has no other source, nor any, where the
   call or another of the callee reads it. It gives the code the
   arguments at the parameters kept ([cc_call], [cc_call_args]) and
   expects the results kept ([call_rebuilt_code_cc]), which what it
   returns to takes directly, or through the wrapper of
   [make_apply_wrapper_cc] ([call_entered_cc]). In the body of the code
   ([code_entered_cc]), its return continuation is masked as the code
   keeps its results: the continuations [rets] of the relation, with the
   kinds of the results, related by [ret_rel], to which the code gives as
   many values as counted, of these kinds, and no call raises
   ([expr_ok]). [ret_rel] is: a way out, where the code keeps every
   result; the handler that the call returns to; the wrapper over the
   handler, which gives poison where the code no longer returns what the
   handler keeps and does not use; the wrapper over a way out, for a call
   in tail position in code that keeps its calling convention, which
   gives poison where the way out has no use; or the wrapper over the
   return continuation of the code that a call in tail position is in,
   recursively ([ret_answered], whose first run does not exit where a
   wrapper is: [ret_rel_no_exit]). A call in tail position in code that
   changes, which keeps what the code keeps, returns to its return
   continuation directly ([ret_rel_transport]). A call in tail position
   in code that changes, of code that does not or of unknown arity,
   returns to the return continuation directly where it keeps every
   result ([ret_rel_all_true]), otherwise through the wrapper of
   [make_apply_wrapper], which is that of [make_apply_wrapper_cc] for
   code that keeps every result ([wrapper_as_cc],
   [ret_rel_plain_wrapper]); the body of the code called is entered with
   its return continuation in [rets], every result kept, unmasked
   ([code_entered_ret]: the third case of [conts_related] takes a
   continuation without a mask as one that keeps every result), and
   [ret_ok] takes such a return continuation for the calls of unknown
   arity of any shape ([returned_either]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaUnusedStaticsSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaFreshQuantifierSpec FlambdaRenameApartSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Simulations in which a step may be answered by none                      *)
(* ======================================================================== *)

Section Shrinking.

Variable R : state -> state -> Prop.
Variable E : list value -> list value -> Prop.
Variable measure : state -> nat.

(* A step of the first state is answered by the second: nothing is asked
   if it is undefined, or if the state it comes to is; an exit, by an
   exit in one step through the same continuation, with the same heap
   and values related by [E]; a step, by one step to a related state,
   or by none, to a state related to the second, if the measure goes
   down. *)
Definition answered (s t : state) : Prop :=
  match step s with
  | Undefined => True
  | Exit k vs hp _ =>
    ∃ (vs' : list value) (rd' : read_record),
      step t = Exit k vs' hp rd' ∧ E vs vs'
  | Step s' =>
    step s' = Undefined
    ∨ (∃ t' : state, step t = Step t' ∧ R s' t')
    ∨ (measure s' < measure s ∧ R s' t)
    ∨ (∃ (t' : state) (n : nat), run (S (S n)) t = Step t' ∧ R s' t')
  end.

Lemma run_prefix (n m : nat) (s t : state) :
  run (n + m) s = Step t -> ∃ t' : state, run n s = Step t'.
Proof.
  rewrite run_add. destruct (run n s) as [t' | | ]; intros H;
    [ exists t'; reflexivity | discriminate H | discriminate H ].
Qed.

Hypothesis R_answered : ∀ s t : state, R s t -> answered s t.

(* An exit of the first run is one of the second, with related values. *)
Lemma shrinking_exit (n : nat) :
  ∀ (s t : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s t -> run n s = Exit k vs hp rd ->
    ∃ (m : nat) (vs' : list value) (rd' : read_record),
      run m t = Exit k vs' hp rd' ∧ E vs vs'.
Proof.
  induction n as [ | n IH]; intros s t k vs hp rd Hrel Hrun;
    [ discriminate Hrun | ].
  rewrite run_succ in Hrun. pose proof (R_answered s t Hrel) as Ha.
  unfold answered in Ha.
  destruct (step s) as [s' | k' vs' hp' rd0 | ] eqn:Hs.
  - destruct Ha as [Hund | [(t' & Ht & Hrel') | [(_ & Hrel')
                                                | (t' & n0 & Ht & Hrel')]]].
    + destruct n as [ | n]; [ discriminate Hrun | ].
      rewrite run_succ, Hund in Hrun. discriminate Hrun.
    + destruct (IH s' t' k vs hp rd Hrel' Hrun) as (m & vs2 & rd2 & Hm & HE).
      exists (S m), vs2, rd2. rewrite run_succ, Ht.
      split; [ exact Hm | exact HE ].
    + exact (IH s' t k vs hp rd Hrel' Hrun).
    + destruct (IH s' t' k vs hp rd Hrel' Hrun) as (m & vs2 & rd2 & Hm & HE).
      exists (S (S n0) + m), vs2, rd2. rewrite run_add, Ht.
      split; [ exact Hm | exact HE ].
  - injection Hrun as <- <- <- <-. destruct Ha as (vs2 & rd2 & Ha & HE).
    exists 1, vs2, rd2. rewrite run_succ, Ha.
    split; [ reflexivity | exact HE ].
  - discriminate Hrun.
Qed.

(* Behaviours of the two runs that correspond: exits through the same
   continuation, with the same heap and values related by [E]; or
   both going wrong. *)
Definition behaviour_related (b b' : behaviour) : Prop :=
  match b, b' with
  | Exits k vs hp, Exits k' vs' hp' => k = k' ∧ hp = hp' ∧ E vs vs'
  | Goes_wrong, Goes_wrong => True
  | _, _ => False
  end.

(* If the second run has ended, the first ends in a corresponding way or
   goes wrong: each step of the first uses up one of the second, or
   lowers the measure. *)
Lemma shrinking_ends (m : nat) :
  ∀ (s t : state) (b : behaviour),
    R s t -> ended (run m t) b ->
    (∃ b0 : behaviour, behaves s b0 ∧ behaviour_related b0 b)
    ∨ behaves s Goes_wrong.
Proof.
  induction m as [m IHm] using (well_founded_induction lt_wf).
  intros s. remember (measure s) as k eqn:Hk. revert s Hk.
  induction k as [k IHk] using (well_founded_induction lt_wf).
  intros s Hk t b Hrel Hended.
  pose proof (R_answered s t Hrel) as Ha. unfold answered in Ha.
  destruct (step s) as [s' | k' vs hp rd | ] eqn:Hs.
  - destruct Ha as [Hund | [(t' & Ht & Hrel') | [(Hlt & Hrel')
                                                | (t' & n0 & Ht & Hrel')]]].
    + right. exists 2. rewrite run_succ, Hs, run_succ, Hund. reflexivity.
    + destruct m as [ | m]; [ destruct b; contradiction | ].
      rewrite run_succ, Ht in Hended.
      destruct (IHm m (Nat.lt_succ_diag_r m) s' t' b Hrel' Hended)
        as [(b0 & Hb & Hbr) | Hw];
        [ left; exists b0; split; [ | exact Hbr ] | right ];
        apply step_behaves; rewrite Hs; assumption.
    + subst k.
      destruct (IHk (measure s') Hlt s' eq_refl t b Hrel' Hended)
        as [(b0 & Hb & Hbr) | Hw];
        [ left; exists b0; split; [ | exact Hbr ] | right ];
        apply step_behaves; rewrite Hs; assumption.
    + destruct (Nat.lt_ge_cases m (S (S n0))) as [Hm | Hm].
      * destruct (run_prefix m (S (S n0) - m) t t') as (t'' & Ht'');
          [ replace (m + (S (S n0) - m)) with (S (S n0)) by lia; exact Ht | ].
        rewrite Ht'' in Hended. destruct b; contradiction.
      * replace m with (S (S n0) + (m - S (S n0))) in Hended by lia.
        rewrite run_add, Ht in Hended.
        destruct (IHm (m - S (S n0)) ltac:(lia) s' t' b Hrel' Hended)
          as [(b0 & Hb & Hbr) | Hw];
          [ left; exists b0; split; [ | exact Hbr ] | right ];
          apply step_behaves; rewrite Hs; assumption.
  - destruct Ha as (vs' & rd' & Ha & HE).
    destruct m as [ | m]; [ destruct b; contradiction | ].
    rewrite run_succ, Ha in Hended. apply ended_exit in Hended. subst b.
    left. exists (Exits k' vs hp). split.
    + apply step_behaves. rewrite Hs. reflexivity.
    + split; [ reflexivity | split; [ reflexivity | exact HE ] ].
  - right. apply step_behaves. rewrite Hs. reflexivity.
Qed.

Lemma shrinking_refines (s t : state) :
  (∀ vs vs' : list value, E vs vs' -> vs' = vs) -> R s t -> refines s t.
Proof.
  intros HE Hrel. split.
  - intros k vs hp [n [rd Hn]].
    destruct (shrinking_exit n s t k vs hp rd Hrel Hn)
      as (m & vs' & rd' & Hm & Hvs).
    rewrite (HE _ _ Hvs) in Hm. exists m, rd'. exact Hm.
  - intros b Hterm Hb. destruct (behaves_ended t b Hterm Hb) as [m Hm].
    destruct (shrinking_ends m s t b Hrel Hm) as [(b0 & Hb0 & Hbr) | Hw];
      [ left | right; exact Hw ].
    destruct b0 as [k vs hp | | ]; destruct b as [k' vs' hp' | | ];
      try contradiction; [ | exact Hb0 ].
    destruct Hbr as (<- & <- & Hvs). rewrite (HE _ _ Hvs). exact Hb0.
Qed.

Lemma shrinking_refines_observably (s t : state) :
  (∀ vs vs' : list value, E vs vs' -> values_related observed_equal vs vs') ->
  R s t -> refines_observably s t.
Proof.
  intros HE Hrel. split; [ | split ].
  - intros k vs hp [n [rd Hn]].
    destruct (shrinking_exit n s t k vs hp rd Hrel Hn)
      as (m & vs' & rd' & Hm & Hvs).
    exists vs'. split; [ exists m, rd'; exact Hm | exact (HE _ _ Hvs) ].
  - intros k vs' hp Hb.
    destruct (behaves_ended t (Exits k vs' hp) ltac:(discriminate) Hb)
      as [m Hm].
    destruct (shrinking_ends m s t _ Hrel Hm) as [(b0 & Hb0 & Hbr) | Hw];
      [ left | right; exact Hw ].
    destruct b0 as [k0 vs hp0 | | ]; try contradiction.
    destruct Hbr as (-> & -> & Hvs). exists vs.
    split; [ exact Hb0 | exact (HE _ _ Hvs) ].
  - intros Hb.
    destruct (behaves_ended t Goes_wrong ltac:(discriminate) Hb) as [m Hm].
    destruct (shrinking_ends m s t _ Hrel Hm) as [(b0 & Hb0 & Hbr) | Hw];
      [ | exact Hw ].
    destruct b0; try contradiction. exact Hb0.
Qed.

End Shrinking.

Lemma values_related_refl (l : list value) :
  Forall (fun v => observed_equal v v) l -> values_related observed_equal l l.
Proof.
  induction 1 as [ | v l Hv _ IH]; simpl; [ exact I | split; assumption ].
Qed.

Lemma observed_equal_refl : ∀ v : value, observed_equal v v.
Proof.
  apply FlambdaHeapProofs.value_induction; cbn [observed_equal].
  - reflexivity.
  - reflexivity.
  - split; reflexivity.
  - reflexivity.
  - intros tag fields IH. split; [ reflexivity | ].
    exact (values_related_refl fields IH).
  - intros elements IH. exact (values_related_refl elements IH).
  - reflexivity.
  - reflexivity.
  - intros callee args arity Hcallee _.
    split; [ exact Hcallee | split; reflexivity ].
Qed.

Lemma values_observed_refl (l : list value) :
  values_related observed_equal l l.
Proof.
  apply values_related_refl, Forall_forall. intros v _.
  apply observed_equal_refl.
Qed.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

(* [P i v v'] of the elements at each position of two lists of one
   length, the first at [i]. *)
Definition fields_related (P : nat -> value -> value -> Prop)
    : nat -> list value -> list value -> Prop :=
  fix fields_related (i : nat) (fields fields' : list value)
      {struct fields} : Prop :=
    match fields, fields' with
    | [], [] => True
    | field :: fields, field' :: fields' =>
      P i field field' ∧ fields_related (S i) fields fields'
    | _, _ => False
    end.

Lemma fields_related_nth (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) (i : nat) (v : value) :
  fields_related P k l l' -> nth_error l i = Some v ->
  ∃ v' : value, nth_error l' i = Some v' ∧ P (k + i) v v'.
Proof.
  revert k l' i. induction l as [ | w l IH]; intros k [ | w' l'] i H Hnth;
    simpl in H; try contradiction; [ destruct i; discriminate | ].
  destruct H as [Hw Hl]. destruct i as [ | i]; simpl in Hnth |- *.
  - injection Hnth as <-. exists w'. rewrite Nat.add_0_r. auto.
  - replace (k + S i) with (S k + i) by lia. exact (IH _ _ _ Hl Hnth).
Qed.

Lemma fields_related_nth' (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) (i : nat) (v' : value) :
  fields_related P k l l' -> nth_error l' i = Some v' ->
  ∃ v : value, nth_error l i = Some v ∧ P (k + i) v v'.
Proof.
  revert k l' i. induction l as [ | w l IH]; intros k [ | w' l'] i H Hnth;
    simpl in H; try contradiction; [ destruct i; discriminate | ].
  destruct H as [Hw Hl]. destruct i as [ | i]; simpl in Hnth |- *.
  - injection Hnth as <-. exists w. rewrite Nat.add_0_r. auto.
  - replace (k + S i) with (S k + i) by lia. exact (IH _ _ _ Hl Hnth).
Qed.

Lemma fields_related_length (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) :
  fields_related P k l l' -> length l' = length l.
Proof.
  revert k l'. induction l as [ | w l IH]; intros k [ | w' l'] H;
    simpl in H; try contradiction; [ reflexivity | ].
  simpl. f_equal. exact (IH _ _ (proj2 H)).
Qed.

Lemma fields_related_intro (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) :
  length l' = length l ->
  (∀ (i : nat) (v v' : value),
     nth_error l i = Some v -> nth_error l' i = Some v' -> P (k + i) v v') ->
  fields_related P k l l'.
Proof.
  revert k l'. induction l as [ | w l IH]; intros k [ | w' l'] Hlen H;
    simpl in Hlen |- *; try discriminate Hlen; [ exact I | ].
  split.
  - specialize (H 0 w w' eq_refl eq_refl). rewrite Nat.add_0_r in H.
    exact H.
  - apply IH; [ lia | ]. intros i v v' Hv Hv'.
    replace (S k + i) with (k + S i) by lia. exact (H (S i) v v' Hv Hv').
Qed.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* Induction on values, with the fields of a block. *)
Lemma value_block_ind (P : value -> Prop) :
  (∀ (tag : Z) (fields : list value),
     (∀ field : value, field ∈ fields -> P field) ->
     P (Value_block tag fields)) ->
  (∀ v : value, (∀ (tag : Z) (fields : list value),
                   v <> Value_block tag fields) -> P v) ->
  ∀ v : value, P v.
Proof.
  intros Hblock Hother. fix IH 1. intros v.
  destruct v as [c | code vs | slot decls vs | s | tag fields | elements | c
                | l | callee args arity];
    try (apply Hother; intros tag' fields' Heq; discriminate Heq).
  apply Hblock. revert fields. fix go 1.
  intros [ | field fields] field' Hin; [ destruct Hin | ].
  destruct Hin as [<- | Hin];
    [ exact (IH field) | exact (go fields field' Hin) ].
Qed.

(* [P w x] of each binding [(w, x)] of a list of value slots that no
   earlier binding shadows: of what [lookup] finds. *)
Definition slots_related (P : value_slot -> value -> Prop)
    : list (value_slot * value) -> list value_slot -> Prop :=
  fix slots_related (vs : list (value_slot * value))
      (seen : list value_slot) {struct vs} : Prop :=
    match vs with
    | [] => True
    | (w, x) :: vs =>
      (among w seen = false -> P w x) ∧ slots_related vs (w :: seen)
    end.

Lemma slots_related_iff (P : value_slot -> value -> Prop)
    (vs : list (value_slot * value)) :
  slots_related P vs []
  <-> ∀ (w : value_slot) (x : value), lookup w vs = Some x -> P w x.
Proof.
  enough (H : ∀ seen : list value_slot,
            slots_related P vs seen
            <-> ∀ (w : value_slot) (x : value),
                  among w seen = false -> lookup w vs = Some x -> P w x)
    by (rewrite H; split; intros H' w x; [ apply H'; reflexivity | ];
        intros _; apply H').
  induction vs as [ | [w' x'] vs IH]; intros seen; simpl.
  - split; [ intros _ w x _ Hl; discriminate Hl | intros _; exact I ].
  - rewrite IH. split.
    + intros [Hw Hrest] w x Hseen Hl.
      destruct (Pos.eqb_spec w w') as [-> | Hne].
      * injection Hl as <-. exact (Hw Hseen).
      * apply (Hrest w x); [ | exact Hl ].
        unfold among in Hseen |- *. simpl.
        rewrite (proj2 (Pos.eqb_neq w w') Hne). exact Hseen.
    + intros H. split.
      * intros Hseen. apply (H w' x' Hseen). rewrite Pos.eqb_refl.
        reflexivity.
      * intros w x Hseen Hl. unfold among in Hseen. simpl in Hseen.
        apply orb_false_iff in Hseen as [Hne Hseen].
        apply (H w x Hseen). rewrite Hne. exact Hl.
Qed.

(* Leaves the cases of [rel] that are not in the fragment. *)
Ltac not_rel :=
  try (match goal with |- False -> _ => intros [] end).

Section Rel.

Variable G : graph.
Variable sol : solution.
Variable kept : value_slot -> bool.
Variable called : node -> code_id -> Prop.
Variable exact : Prop.
(* Whether the runs may have partial applications. *)
Variable partials : Prop.

(* What the second run has for a value of the first at a node with any
   usage: the same, but for the value slots of the closures in it, which
   are the same if [exact]; each symbol in it has any usage. *)
Fixpoint sim (v v' : value) {struct v} : Prop :=
  match v with
  | Value_const _ | Value_boxed_number _ => v' = v
  | Value_symbol s => v' = v ∧ s_any_usage sol (Node_symbol s) = true
  | Value_block tag fields =>
    ∃ fields' : list value,
      v' = Value_block tag fields'
      ∧ values_related (fun w w' => sim w w') fields fields'
  | Value_immutable_array elements =>
    ∃ elements' : list value,
      v' = Value_immutable_array elements'
      ∧ values_related (fun w w' => sim w w') elements elements'
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ value_slots' : list (value_slot * value),
      v' = Value_closure_of_set slot function_decls value_slots'
      ∧ (exact -> value_slots' = value_slots)
  | Value_partial_application closure args args_arity =>
    partials
    ∧ ∃ (closure' : value) (args' : list value),
        v' = Value_partial_application closure' args' args_arity
        ∧ (exact -> v' = v)
        ∧ observed_equal closure closure'
        ∧ map kind_of_value args' = map kind_of_value args
  | Value_closure _ _ | Value_location _ => False
  end.

(* The witness of unknown arity that a call of what may come from an
   allocation of [P] reaches with its group [g] of arguments: the call
   witness of unknown arity of the allocation, then that of an
   allocation that the first result of the witness before may come from,
   as the code has them ([code_chain_edges]) and as the rebuild reads
   them ([arguments_used_by_call]); each is an allocation of its own. *)
Fixpoint witness_chain (P : node -> Prop) (g : nat) (W : node) : Prop :=
  match g with
  | O =>
    ∃ y : node,
      P y
      ∧ g_constructor G y (Field_call_witness Unknown_arity_code_pointer) W
      ∧ may_come_from sol W W
  | S g =>
    ∃ W' R v : node,
      witness_chain P g W'
      ∧ g_constructor G W' (Field_return_of_call (Normal 0)) R
      ∧ may_come_from sol v R
      ∧ may_come_from sol v v
      ∧ g_constructor G v (Field_call_witness Unknown_arity_code_pointer) W
      ∧ may_come_from sol W W
  end.

(* The value [v] of the first run may be at the node [n], and [v'] is
   what the second run has for it. A constant is the same in the second
   run, at a node with any source; the poison of the kinds [Region] and
   [Rec_info], which is the only value of those kinds, may be at any
   node, as the rebuild gives it for an argument of those kinds at a
   parameter that has no use, which the code keeps. A block has in the
   second run its tag, each field used related, and poison in the
   others. A closure has the nodes of the closures of its set ([sites])
   as FlambdaReaperFlowSpec.flows has them; in the second run it is of
   the same set, with the value slots that the rebuild keeps ([exact]:
   all), and a value slot kept and used at one of these nodes is related
   there. The elements of an immutable array are related at nodes with
   any usage. A partial application, if the runs may have any, has its
   closure related at some node [m] that has a use, and observed equal,
   and an arity of one parameter at least; what the last of the
   witnesses that its groups reach from the allocations that [m] may
   come from returns first may come to [n]; and it has in the second run
   arguments of the same kinds, each related at the parameters that have
   a use of the witnesses that its group reaches: those that the code of
   the closure reads. *)
Fixpoint rel (v : value) (n : node) (v' : value) {struct v} : Prop :=
  match v with
  | Value_const c =>
    (argument_kind_ok (kind_of_const c) = true -> s_any_source sol n = true)
    ∧ v' = v
  | Value_block tag fields =>
    ∃ (a : node) (fields' : list value),
      may_come_from sol a n
      ∧ v' = Value_block tag fields'
      ∧ fields_related
          (fun i field field' =>
             ∃ from : node,
               g_constructor G a (Field_block i Value) from
               ∧ (s_field_used sol a (Field_block i Value) = true ->
                  rel field from field')
               ∧ (s_field_used sol a (Field_block i Value) = false ->
                  field' = Value_const (Poison Value)))
          0 fields fields'
  | Value_boxed_number c =>
    ∃ (a : node) (bn : boxable_number) (from : node),
      may_come_from sol a n
      ∧ is_number_of_kind bn c = true
      ∧ g_constructor G a (Field_boxed_number bn) from
      ∧ v' = (if s_field_used sol a (Field_boxed_number bn) then v
              else Value_const (Poison Value))
  | Value_symbol s =>
    v' = v ∧ may_come_from sol (Node_symbol s) n
    ∧ (s_has_usage sol n = true -> s_has_usage sol (Node_symbol s) = true)
  | Value_immutable_array elements =>
    s_any_source sol n = true
    ∧ ∃ elements' : list value,
        v' = Value_immutable_array elements'
        ∧ values_related
            (fun w w' =>
               (w' = w
                ∧ ((∃ c : const, w = Value_const c)
                   ∨ ∃ s : symbol,
                       w = Value_symbol s
                       ∧ s_any_usage sol (Node_symbol s) = true))
               ∨ ∃ m : node, s_any_usage sol m = true ∧ rel w m w')
            elements elements'
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ (sites : function_slot -> node)
      (value_slots' : list (value_slot * value)),
      may_come_from sol (sites slot) n
      ∧ v' = Value_closure_of_set slot function_decls value_slots'
      ∧ (exact -> value_slots' = value_slots)
      ∧ ∀ (f : function_slot) (id : code_id),
          lookup f function_decls = Some id ->
          may_come_from sol (sites f) (sites f)
          ∧ called (sites f) id
          ∧ (∀ (g : function_slot) (id' : code_id),
               lookup g function_decls = Some id' ->
               g_constructor G (sites f) (Field_function_slot g) (sites g))
          ∧ slots_related
              (fun w x =>
                 kept w = true ->
                 ∃ from : node,
                   g_constructor G (sites f) (Field_value_slot w) from
                   ∧ (s_field_used sol (sites f) (Field_value_slot w) = true ->
                      ∃ x' : value,
                        lookup w value_slots' = Some x'
                        ∧ (s_has_usage sol from = true -> rel x from x')))
              value_slots []
  | Value_partial_application closure args args_arity =>
    partials
    ∧ ∃ (m : node) (closure' : value) (args' : list value),
        v' = Value_partial_application closure' args' args_arity
        ∧ (exact -> v' = v)
        ∧ rel closure m closure'
        ∧ s_has_usage sol m = true
        ∧ observed_equal closure closure'
        ∧ args_arity <> []
        ∧ (∀ (g : nat) (W R y : node),
             S g = length args_arity ->
             witness_chain
               (fun y => may_come_from sol y m ∧ may_come_from sol y y) g W ->
             g_constructor G W (Field_return_of_call (Normal 0)) R ->
             may_come_from sol y R -> may_come_from sol y y ->
             may_come_from sol y n)
        ∧ map kind_of_value args' = map kind_of_value args
        ∧ fields_related
            (fun i arg arg' =>
               ∀ (g k : nat) (W p : node),
                 group_position (arity_sizes args_arity) i = Some (g, k) ->
                 witness_chain
                   (fun y => may_come_from sol y m ∧ may_come_from sol y y)
                   g W ->
                 g_parameter G W k p -> s_has_usage sol p = true ->
                 rel arg p arg')
            0 args args'
  | Value_closure _ _ | Value_location _ => False
  end.

(* A field of a block that may come from [a]. *)
Definition field_rel (a : node) (i : nat) (field field' : value) : Prop :=
  ∃ from : node,
    g_constructor G a (Field_block i Value) from
    ∧ (s_field_used sol a (Field_block i Value) = true ->
       rel field from field')
    ∧ (s_field_used sol a (Field_block i Value) = false ->
       field' = Value_const (Poison Value)).

(* A value slot [w] of a closure that may come from [a], with the value
   slots [value_slots'] in the second run. *)
Definition slot_rel (a : node) (value_slots' : list (value_slot * value))
    (w : value_slot) (x : value) : Prop :=
  kept w = true ->
  ∃ from : node,
    g_constructor G a (Field_value_slot w) from
    ∧ (s_field_used sol a (Field_value_slot w) = true ->
       ∃ x' : value,
         lookup w value_slots' = Some x'
         ∧ (s_has_usage sol from = true -> rel x from x')).

Lemma rel_kind (v : value) (n : node) (v' : value) :
  rel v n v' -> kind_of_value v' = kind_of_value v.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ ->]. reflexivity.
  - intros (sites & vs' & _ & -> & _). reflexivity.
  - intros [-> _]. reflexivity.
  - intros (a & fields' & _ & -> & _). reflexivity.
  - intros (_ & elements' & -> & _). reflexivity.
  - intros (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - intros [_ (m & closure' & args' & -> & _)]. reflexivity.
Qed.

(* No value is related to a location. *)
Lemma rel_not_location (v : value) (n : node) (v' : value) (l : location) :
  rel v n v' -> v' <> Value_location l.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ ->]. discriminate.
  - intros (sites & vs' & _ & -> & _). discriminate.
  - intros [-> _]. discriminate.
  - intros (a & fields' & _ & -> & _). discriminate.
  - intros (_ & elements' & -> & _). discriminate.
  - intros (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate.
  - intros [_ (m & closure' & args' & -> & _)]. discriminate.
Qed.

Lemma rel_tagged (v : value) (n : node) (v' : value) :
  rel v n v' -> is_tagged_immediate v' = is_tagged_immediate v.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ ->]. reflexivity.
  - intros (sites & vs' & _ & -> & _). reflexivity.
  - intros [-> _]. reflexivity.
  - intros (a & fields' & _ & -> & _). reflexivity.
  - intros (_ & elements' & -> & _). reflexivity.
  - intros (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - intros [_ (m & closure' & args' & -> & _)]. reflexivity.
Qed.

(* What the second run has is not a symbol unless the first has it. *)
Lemma rel_symbol (v : value) (n : node) (s : symbol) :
  rel v n (Value_symbol s) -> v = Value_symbol s.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ Heq]. discriminate Heq.
  - intros (sites & vs' & _ & Heq & _). discriminate Heq.
  - intros [Heq _]. exact (eq_sym Heq).
  - intros (a & fields' & _ & Heq & _). discriminate Heq.
  - intros (_ & elements' & Heq & _). discriminate Heq.
  - intros (a & bn & from & _ & _ & _ & Heq).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate Heq.
  - intros [_ (m & closure' & args' & Heq & _)]. discriminate Heq.
Qed.

(* What [sim] relates is observed equal, and equal if [exact]. *)
Lemma sims_observed (ws : list value) :
  Forall (fun w => ∀ w' : value,
            sim w w' -> observed_equal w w' ∧ (exact -> w' = w)) ws ->
  ∀ ws' : list value,
    values_related (fun w w' => sim w w') ws ws' ->
    values_related observed_equal ws ws' ∧ (exact -> ws' = ws).
Proof.
  induction 1 as [ | w ws Hw _ IH]; intros [ | w' ws'] Hws; simpl in Hws |- *;
    try contradiction; [ split; [ exact I | reflexivity ] | ].
  destruct Hws as [Hw' Hrest].
  destruct (Hw w' Hw') as [Ho He]. destruct (IH ws' Hrest) as [Ho' He'].
  split; [ split; assumption | ].
  intros Hx. rewrite (He Hx), (He' Hx). reflexivity.
Qed.

Lemma sim_observed : ∀ v v' : value,
  sim v v' -> observed_equal v v' ∧ (exact -> v' = v).
Proof.
  apply (FlambdaHeapProofs.value_induction
           (fun v => ∀ v' : value,
              sim v v' -> observed_equal v v' ∧ (exact -> v' = v)));
    cbn [sim].
  - intros c v' ->. split; [ apply observed_equal_refl | reflexivity ].
  - intros code slots _ v' [].
  - intros slot decls vs _ v' (vs' & -> & Hex).
    split; [ cbn [observed_equal]; auto | ].
    intros He. rewrite (Hex He). reflexivity.
  - intros s v' [-> _]. split; [ apply observed_equal_refl | reflexivity ].
  - intros tag fields IH v' (fields' & -> & Hf).
    destruct (sims_observed fields IH fields' Hf) as [Ho He].
    split; [ cbn [observed_equal]; split; [ reflexivity | exact Ho ] | ].
    intros Hx. rewrite (He Hx). reflexivity.
  - intros elements IH v' (elements' & -> & Hf).
    destruct (sims_observed elements IH elements' Hf) as [Ho He].
    split; [ exact Ho | intros Hx; rewrite (He Hx); reflexivity ].
  - intros c v' ->. split; [ apply observed_equal_refl | reflexivity ].
  - intros l v' [].
  - intros callee args arity _ _ v' [_ (closure' & args' & -> & Hex & Ho & Hk)].
    split; [ | exact Hex ].
    cbn [observed_equal].
    split; [ exact Ho | split; [ symmetry; exact Hk | reflexivity ] ].
Qed.

Variable is_local : field -> bool.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.

Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hs n := (s_has_source sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).

Lemma block_field_not_local (i : nat) (kind : flambda_kind) :
  is_local (Field_block i kind) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

Lemma boxed_field_not_local (bn : boxable_number) :
  is_local (Field_boxed_number bn) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

(* Along an alias of the solution. *)
Lemma rel_alias (v : value) (to_ from : node) (v' : value) :
  s_alias sol to_ from = true -> rel v from v' -> rel v to_ v'.
Proof.
  intros Hal. destruct v; cbn [rel]; not_rel.
  - intros [Has ->]. split; [ | reflexivity ].
    intros Hk. exact (any_source_alias _ _ _ Hc _ _ Hal (Has Hk)).
  - intros (sites & vs' & Ha & Hrest). exists sites, vs'.
    split; [ exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha) | exact Hrest ].
  - intros (-> & Ha & Hhu). split; [ reflexivity | split ].
    + exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha).
    + intros Hto. apply Hhu. exact (has_usage_alias _ _ _ Hc _ _ Hto Hal).
  - intros (a & fields' & Ha & Hrest). exists a, fields'.
    split; [ exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha) | exact Hrest ].
  - intros [Has Hrest]. split; [ | exact Hrest ].
    exact (any_source_alias _ _ _ Hc _ _ Hal Has).
  - intros (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha) | exact Hrest ].
  - intros [Hp (m & closure' & args' & Hv' & Hex & Hcl & Hhm & Ho & Hne
                 & Hend & Hrest)].
    split; [ exact Hp | ]. exists m, closure', args'.
    do 6 (split; [ assumption | ]). split; [ | exact Hrest ].
    intros g W R y Hg HW HR HyR Hyy.
    exact (may_come_from_alias _ _ _ Hc _ _ _ Hal
             (Hend g W R y Hg HW HR HyR Hyy)).
Qed.

(* To a node with any source that has a usage only if [m] has any
   usage. *)
Lemma rel_escape (v : value) (to_ m : node) (v' : value) :
  as_ to_ -> (hu to_ -> au m) -> rel v m v' -> rel v to_ v'.
Proof.
  intros Has Hm. destruct v; cbn [rel]; not_rel.
  - intros [_ ->]. split; [ intros _; exact Has | reflexivity ].
  - intros (sites & vs' & Ha & Hrest). exists sites, vs'.
    split; [ exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha)
           | exact Hrest ].
  - intros (-> & Ha & _). split; [ reflexivity | split ].
    + exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha).
    + intros Hto. apply (au_hu _ _ _ Hc). destruct Ha as (_ & _ & _ & Hau & _).
      apply Hau, Hm, Hto.
  - intros (a & fields' & Ha & Hrest). exists a, fields'.
    split; [ exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha)
           | exact Hrest ].
  - intros [_ Hrest]. split; [ exact Has | exact Hrest ].
  - intros (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha)
           | exact Hrest ].
  - intros [Hp (m' & closure' & args' & Hv' & Hex & Hcl & Hhm & Ho & Hne
                 & Hend & Hrest)].
    split; [ exact Hp | ]. exists m', closure', args'.
    do 6 (split; [ assumption | ]). split; [ | exact Hrest ].
    intros g W R y Hg HW HR HyR Hyy.
    exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm
             (Hend g W R y Hg HW HR HyR Hyy)).
Qed.

(* The read of the field [f] of a value that may come from [a], as
   FlambdaReaperFlowProofs.flows_read. *)
Lemma rel_read (a base to_ from : node) (f : field) (v v' : value) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  rel v from v' -> rel v to_ v'.
Proof.
  intros Hl Hcons Hacc (_ & Hsrc & _ & Hau & Hus) Hv.
  destruct (s_any_source sol base) eqn:Eb.
  - pose proof (any_source_accessor _ _ _ Hc _ _ _ Eb Hacc Hl) as Has.
    destruct (s_any_usage sol a) eqn:Ea.
    + apply (rel_escape v to_ from v' Has); [ | exact Hv ].
      intros _. exact (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hl).
    + destruct (s_has_usage sol to_) eqn:Ehu.
      * pose proof (has_usage_accessor _ _ _ Hc _ _ _ Ehu Hacc) as Hhu.
        assert (Eau : s_any_usage sol base = false).
        { destruct (s_any_usage sol base);
          [ discriminate (Hau eq_refl) | reflexivity ]. }
        pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
        destruct (Hus Hhu base Hbb) as [Hab | Hab]; [ | congruence ].
        apply (rel_alias v to_ from v'); [ | exact Hv ].
        exact (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Ea Hacc).
      * apply (rel_escape v to_ from v' Has); [ | exact Hv ].
        intros Hhu. congruence.
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    apply (rel_alias v to_ from v'); [ | exact Hv ].
    exact (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hacc Hsrc Eb Hcons).
Qed.

(* At a node with any usage, what the second run has is [sim]. *)
Lemma rel_au_sim : ∀ (v : value) (n : node) (v' : value),
  au n -> rel v n v' -> sim v v'.
Proof.
  apply (FlambdaHeapProofs.value_induction
           (fun v => ∀ (n : node) (v' : value),
              au n -> rel v n v' -> sim v v'));
    cbn [rel sim].
  - intros c n v' _ [_ ->]. reflexivity.
  - intros code slots _ n v' _ [].
  - intros slot decls vs _ n v' _ (sites & vs' & _ & -> & Hex & _).
    exists vs'. split; [ reflexivity | exact Hex ].
  - intros s n v' Hn (-> & Ha & _). split; [ reflexivity | ].
    destruct Ha as (_ & _ & _ & Hau & _). exact (Hau Hn).
  - intros tag fields IH n v' Hn (a & fields' & Ha & -> & Hfields).
    exists fields'. split; [ reflexivity | ].
    destruct Ha as (_ & _ & _ & Hau & _). specialize (Hau Hn).
    revert fields' Hfields. generalize 0 as k.
    induction IH as [ | field fields Hfield _ IHf]; intros k [ | field' fields']
      Hfields; simpl in Hfields |- *; try contradiction; [ exact I | ].
    destruct Hfields as [(from & Hcons & Hused & _) Hrest]. split.
    + pose proof (block_field_not_local k Value) as Hl.
      apply (Hfield from);
        [ exact (any_usage_constructor _ _ _ Hc _ _ _ Hau Hcons Hl) | ].
      apply Hused. exact (field_used_any_usage _ _ _ Hc _ _ _ Hcons Hau Hl).
    + exact (IHf (S k) fields' Hrest).
  - intros elements IH n v' _ (_ & elements' & -> & Helements).
    exists elements'. split; [ reflexivity | ].
    revert elements' Helements.
    induction IH as [ | w elements Hw _ IHe]; intros [ | w' elements']
      Helements; simpl in Helements |- *; try contradiction; [ exact I | ].
    destruct Helements as [Hel Hrest]. split; [ | exact (IHe elements' Hrest) ].
    destruct Hel as [(-> & [(c & ->) | (s & -> & Hs)]) | (m & Hm & Hrel)].
    + reflexivity.
    + split; [ reflexivity | exact Hs ].
    + exact (Hw m w' Hm Hrel).
  - intros c n v' Hn (a & bn & from & (_ & _ & _ & Hau & _) & _ & Hcons & ->).
    rewrite (field_used_any_usage _ _ _ Hc _ _ _ Hcons (Hau Hn)
               (boxed_field_not_local bn)).
    reflexivity.
  - intros l n v' _ [].
  - intros callee args arity _ _ n v' _
      [Hp (m & closure' & args' & -> & Hex & _ & _ & Ho & _ & _ & Hk & _)].
    split; [ exact Hp | ]. exists closure', args'.
    split; [ reflexivity | split; [ exact Hex | split; assumption ] ].
Qed.

Hypothesis Hexact : field_used_exact is_local G sol.

(* A field that is used holds what has a use, whichever of its
   constructor facts. *)
Lemma field_used_hu (a from : node) (f : field) :
  is_local f = false ->
  s_field_used sol a f = true -> g_constructor G a f from -> hu from.
Proof.
  intros Hl Hused Hcons.
  destruct (s_any_usage sol a) eqn:Ea.
  - exact (au_hu _ _ _ Hc _ (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hl)).
  - destruct (Hexact a f Hused)
      as (from0 & _ & [[Hau _] | [(usage & v & Hus & Hacc & Hhu)
                                | [(x & Hau & _) | (_ & _ & _ & Hl' & _)]]]);
      [ congruence | | congruence | congruence ].
    apply (has_usage_alias _ _ _ Hc v from Hhu).
    exact (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hus Ea Hacc).
Qed.

End Rel.

(* ======================================================================== *)
(* What the unit asks of an expression                                      *)
(* ======================================================================== *)

(* The [Switch]es of the expression, of its handlers and of its code are
   on what has any usage, as the traversal makes them ([cond_any_usage]),
   or in code on what the code id uses. *)
Fixpoint scrutinees_used (G : graph) (ctx : option code_id) (e : expr)
    : Prop :=
  match e with
  | Let _ _ _ body => scrutinees_used G ctx body
  | Let_static bindings body =>
    every (binding_scrutinees_used G) bindings ∧ scrutinees_used G ctx body
  | Let_cont_non_recursive _ h body =>
    handler_scrutinees_used G ctx h ∧ scrutinees_used G ctx body
  | Let_cont_recursive handlers body =>
    every (fun '(_, h) => handler_scrutinees_used G ctx h) handlers
    ∧ scrutinees_used G ctx body
  | Apply _ | Apply_cont _ | Invalid => True
  | Switch scrutinee _ =>
    match ctx with
    | None => g_any_usage G (node_of_simple scrutinee)
    | Some id => g_use G (Node_code_id id) (node_of_simple scrutinee)
    end
  end
with handler_scrutinees_used (G : graph) (ctx : option code_id)
    (h : continuation_handler) : Prop :=
  match h with
  | Handler _ _ handler => scrutinees_used G ctx handler
  end
with binding_scrutinees_used (G : graph) (binding : static_binding)
    : Prop :=
  match binding with
  | Bind_code id c => code_scrutinees_used G id c
  | Bind_deleted_code _ | Bind_symbol _ _ => True
  end
with code_scrutinees_used (G : graph) (id : code_id) (c : code) : Prop :=
  match c with
  | Code _ _ _ _ _ body => scrutinees_used G (Some id) body
  end.

(* A [Let] that is not of a set of closures. *)
Definition named_plain (d : named) : bool :=
  match d with
  | Set_of_closures _ _ | Closure_of_set _ _ _ => false
  | _ => true
  end.

(* The check of the units of the proof: [no_code_no_slots] if the
   rebuild is compared exactly ([exact]); otherwise [no_call], with code
   whose body is of the check, the static constants of
   [static_constants], and any call (where calls go is [exits_bound]
   and [raises_kept]; what they call, [in_flow_fragment]). [codes] is
   not read. *)
Definition named_checked (exact : bool) (kind : flambda_kind) (d : named)
    : bool :=
  if exact then named_no_code_no_slots d else named_no_code kind d.

Fixpoint checked (exact : bool) (codes : list (code_id * code)) (e : expr)
    : bool :=
  match e with
  | Let _ kind d body => named_checked exact kind d && checked exact codes body
  | Let_static bindings body =>
    forallb (binding_checked exact codes) bindings && checked exact codes body
  | Let_cont_non_recursive _ h body =>
    handler_checked exact codes h && checked exact codes body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_checked exact codes h) handlers
    && checked exact codes body
  | Apply _ => negb exact
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_checked (exact : bool) (codes : list (code_id * code))
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => checked exact codes handler
  end
with binding_checked (exact : bool) (codes : list (code_id * code))
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => negb exact && code_checked exact codes c
  | Bind_deleted_code _ | Bind_symbol _ _ =>
    if exact then binding_no_code_no_slots binding
    else binding_static_constant binding
  end
with code_checked (exact : bool) (codes : list (code_id * code)) (c : code)
    : bool :=
  match c with
  | Code _ _ _ _ _ body => checked exact codes body
  end.

Lemma named_checked_no_code (exact : bool) (kind : flambda_kind) (d : named) :
  named_checked exact kind d = true -> named_no_code kind d = true.
Proof.
  intros H. destruct exact; [ | exact H ].
  destruct d; simpl in H |- *; try discriminate H; reflexivity.
Qed.

(* A binding of the check: of no code, or of code whose body is of the
   check, up to observation. *)
Lemma binding_checked_cases (exact : bool) (codes : list (code_id * code))
    (b : static_binding) :
  binding_checked exact codes b = true ->
  binding_static_constant b = true
  ∨ ∃ (id : code_id) (c : code),
      b = Bind_code id c ∧ exact = false
      ∧ checked false codes (code_body c) = true.
Proof.
  intros H. destruct exact.
  - left. destruct b as [ | | s [ | slot syms decls [ | ] | | | | ]];
      simpl in H |- *; try discriminate H; try reflexivity; exact H.
  - destruct b as [id c | id | s const]; simpl in H.
    + right. exists id, c. destruct c. auto.
    + left. reflexivity.
    + left. exact H.
Qed.

(* A unit without code meets [no_call]. *)
Lemma no_code_no_call_all :
  (∀ e : expr, no_code e = true -> no_call e = true)
  ∧ (∀ h : continuation_handler,
       handler_no_code h = true -> handler_no_call h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd, (IH Hb). reflexivity.
  - intros bindings body _ IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ discriminate Hbs | reflexivity | exact Hbs ].
  - intros k h body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - intros ac H. exact H.
  - intros scrutinee arms H. exact H.
  - intros H. exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* The [Apply_cont]s of continuations that no handler in scope ([bound])
   binds give as many values as [ce] has nodes for them: one to the
   continuations of the unit, and to those of the code they are in as many
   as counted; and a call that does not return, or raise, to a handler in
   scope gives as many values as [ce] has nodes for its continuation, or
   for its exception continuation (the exception and the extra
   arguments). *)
Definition count_is (ce : finite_map (list node)) (k : continuation)
    (n : nat) : bool :=
  match lookup k ce with
  | Some ns => Nat.eqb (length ns) n
  | None => true
  end.

Definition exits_ac (bound : list continuation) (ce : finite_map (list node))
    (ac : apply_cont) : bool :=
  among (ac_continuation ac) bound
  || count_is ce (ac_continuation ac) (length (ac_args ac)).

Fixpoint exits_bound (bound : list continuation)
    (ce : finite_map (list node)) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => exits_bound bound ce body
  | Let_cont_non_recursive k h body =>
    handler_exits_bound bound ce h
    && exits_bound (k :: bound) ((k, handler_params h) :: ce) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_exits_bound (map fst handlers ++ bound)
           (rec_conts ce handlers) h)
      handlers
    && exits_bound (map fst handlers ++ bound) (rec_conts ce handlers) body
  | Apply ap =>
    (among (ap_exn_continuation ap) bound
     || count_is ce (ap_exn_continuation ap)
          (S (length (ap_exn_extra_args ap))))
    && (among (ap_continuation ap) bound
        || count_is ce (ap_continuation ap)
             (length (unarize (ap_return_arity ap))))
  | Invalid => true
  | Apply_cont ac => exits_ac bound ce ac
  | Switch _ arms => forallb (fun '(_, ac) => exits_ac bound ce ac) arms
  end
with handler_exits_bound (bound : list continuation)
    (ce : finite_map (list node)) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => exits_bound bound ce handler
  end.

(* In code, [Apply_cont]s give as many values as counted. *)
Lemma exits_bound_of_counted :
  (∀ (e : expr) (ks : finite_map nat) (bound : list continuation)
     (ce : finite_map (list node)),
     conts_counted ks e = true ->
     (∀ (k : continuation) (m : nat),
        lookup k ks = Some m -> among k bound = false ->
        ∃ ns : list node, lookup k ce = Some ns ∧ length ns = m) ->
     exits_bound bound ce e = true)
  ∧ (∀ (h : continuation_handler) (ks : finite_map nat)
       (bound : list continuation) (ce : finite_map (list node)),
       handler_conts_counted ks h = true ->
       (∀ (k : continuation) (m : nat),
          lookup k ks = Some m -> among k bound = false ->
          ∃ ns : list node, lookup k ce = Some ns ∧ length ns = m) ->
       handler_exits_bound bound ce h = true).
Proof.
  assert (Hcount : ∀ (k : continuation) (n : nat) (ks : finite_map nat)
                     (bound : list continuation)
                     (ce : finite_map (list node)),
            count_ok ks k n = true ->
            (∀ (k : continuation) (m : nat),
               lookup k ks = Some m -> among k bound = false ->
               ∃ ns : list node, lookup k ce = Some ns ∧ length ns = m) ->
            among k bound || count_is ce k n = true).
  { intros k n ks bound ce Hc Hks.
    destruct (among k bound) eqn:Hb; [ reflexivity | ].
    unfold count_ok in Hc.
    destruct (lookup k ks) as [m | ] eqn:Hl; [ | discriminate Hc ].
    apply Nat.eqb_eq in Hc. subst m.
    destruct (Hks k _ Hl Hb) as (ns & Hns & Hlen).
    unfold count_is. rewrite Hns, Hlen, Nat.eqb_refl. reflexivity. }
  apply expr_handler_ind.
  - intros x kind d body IH ks bound ce Hc Hks. exact (IH ks bound ce Hc Hks).
  - intros bindings body _ IH ks bound ce Hc Hks. simpl in Hc |- *.
    exact (IH ks bound ce Hc Hks).
  - intros k0 h body IHh IH ks bound ce Hc Hks. simpl in Hc |- *.
    apply andb_true_iff in Hc as [Hch Hcb].
    rewrite (IHh ks bound ce Hch Hks). simpl.
    apply (IH _ _ _ Hcb). intros k m Hl Hb.
    rewrite among_cons in Hb. apply orb_false_iff in Hb as [Hk Hb].
    simpl in Hl |- *. rewrite Hk in Hl |- *. exact (Hks k m Hl Hb).
  - intros handlers body IHh IH ks bound ce Hc Hks. simpl in Hc |- *.
    apply andb_true_iff in Hc as [Hch Hcb].
    rewrite forallb_forall in Hch.
    assert (Hks' : ∀ (k : continuation) (m : nat),
              lookup k
                (map (fun '(k, h) => (k, length (handler_params h))) handlers
                 ++ ks)
              = Some m ->
              among k (map fst handlers ++ bound) = false ->
              ∃ ns : list node,
                lookup k (rec_conts ce handlers) = Some ns
                ∧ length ns = m).
    { intros k m Hl Hb. rewrite among_app in Hb.
      apply orb_false_iff in Hb as [Hk Hb]. rewrite lookup_app in Hl.
      unfold rec_conts. rewrite lookup_app.
      assert (Hnone : ∀ (B : Type) (f : continuation_handler -> B),
                lookup k (map (fun '(k, h) => (k, f h)) handlers) = None).
      { intros B f.
        destruct (lookup k (map (fun '(k, h) => (k, f h)) handlers))
          as [m' | ] eqn:Hl'; [ | reflexivity ].
        exfalso. apply lookup_in in Hl'. apply in_map_iff in Hl'
          as ([k' h'] & [= <- _] & Hin).
        apply not_true_iff_false in Hk. apply Hk. apply among_In.
        exact (in_map fst _ (k', h') Hin). }
      pose proof (Hnone _ (fun h => length (handler_params h))) as E1.
      cbv beta in E1. rewrite E1 in Hl.
      rewrite (Hnone _ handler_params). exact (Hks k m Hl Hb). }
    apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hin.
      exact (IHh k h Hin _ _ _ (Hch _ Hin) Hks').
    + exact (IH _ _ _ Hcb Hks').
  - intros ap ks bound ce Hc Hks. simpl in Hc |- *.
    apply andb_true_iff in Hc as [Hret Hexn_c].
    apply andb_true_intro.
    split; [ exact (Hcount _ _ _ _ _ Hexn_c Hks)
           | exact (Hcount _ _ _ _ _ Hret Hks) ].
  - intros ac ks bound ce Hc Hks. exact (Hcount _ _ _ _ _ Hc Hks).
  - intros scrutinee arms ks bound ce Hc Hks. simpl in Hc |- *.
    rewrite forallb_forall in Hc |- *. intros [i ac] Hin.
    exact (Hcount _ _ _ _ _ (Hc _ Hin) Hks).
  - reflexivity.
  - intros params metadata handler IH ks bound ce Hc Hks.
    exact (IH ks bound ce Hc Hks).
Qed.

(* Each call that returns to a handler in scope ([ks]: the kinds of its
   parameters) returns values of these kinds, in code too. *)
Fixpoint calls_kinded (ks : finite_map (list flambda_kind)) (e : expr)
    : bool :=
  match e with
  | Let _ _ _ body => calls_kinded ks body
  | Let_static bindings body =>
    forallb binding_calls_kinded bindings && calls_kinded ks body
  | Let_cont_non_recursive k h body =>
    handler_calls_kinded ks h
    && calls_kinded ((k, handler_kinds h) :: ks) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_calls_kinded
           (map (fun '(k, h) => (k, handler_kinds h)) handlers ++ ks) h)
      handlers
    && calls_kinded
         (map (fun '(k, h) => (k, handler_kinds h)) handlers ++ ks) body
  | Apply ap =>
    match lookup (ap_continuation ap) ks with
    | Some expected => kinds_eqb expected (unarize (ap_return_arity ap))
    | None => true
    end
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_calls_kinded (ks : finite_map (list flambda_kind))
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => calls_kinded ks handler
  end
with binding_calls_kinded (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_calls_kinded c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_calls_kinded (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => calls_kinded [] body
  end.

(* Each call that raises to a handler in scope ([xs]: whether it is an
   exception handler of a non-recursive [Let_cont]) raises to an
   exception handler, and each call that raises to another continuation
   gives it no extra argument, in code too. *)
Fixpoint raises_kept (xs : finite_map bool) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => raises_kept xs body
  | Let_static bindings body =>
    forallb binding_raises_kept bindings && raises_kept xs body
  | Let_cont_non_recursive k h body =>
    handler_raises_kept xs h
    && raises_kept ((k, handler_is_exn_handler h) :: xs) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_raises_kept (map (fun '(k, _) => (k, false)) handlers ++ xs)
           h)
      handlers
    && raises_kept (map (fun '(k, _) => (k, false)) handlers ++ xs) body
  | Apply ap =>
    match lookup (ap_exn_continuation ap) xs with
    | Some b => b
    | None => no_exn_extra_args ap
    end
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_raises_kept (xs : finite_map bool) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => raises_kept xs handler
  end
with binding_raises_kept (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_raises_kept c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_raises_kept (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => raises_kept [] body
  end.

(* A continuation that no call names adds nothing to check of calls. *)
Lemma calls_kinded_unused (r : continuation) (kinds : list flambda_kind) :
  (∀ e : expr,
     ∀ ks : finite_map (list flambda_kind),
       no_call_returns_to r e = true -> calls_kinded ks e = true ->
       calls_kinded (ks ++ [(r, kinds)]) e = true)
  ∧ (∀ h : continuation_handler,
       ∀ ks : finite_map (list flambda_kind),
         handler_no_call_returns_to r h = true ->
         handler_calls_kinded ks h = true ->
         handler_calls_kinded (ks ++ [(r, kinds)]) h = true).
Proof.
  apply expr_handler_ind; try (intros; reflexivity).
  - intros x kind d body IH ks Hn Hk. exact (IH ks Hn Hk).
  - intros bindings body _ IH ks Hn Hk. cbn [calls_kinded] in Hk |- *.
    apply andb_true_iff in Hk as [Hb Hk]. rewrite Hb.
    exact (IH ks Hn Hk).
  - intros k h body IHh IH ks Hn Hk. cbn [calls_kinded] in Hk |- *.
    cbn [no_call_returns_to] in Hn.
    apply andb_true_iff in Hn as [Hnh Hnb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite (IHh ks Hnh Hkh).
    exact (IH ((k, handler_kinds h) :: ks) Hnb Hkb).
  - intros handlers body IHh IH ks Hn Hk. cbn [calls_kinded] in Hk |- *.
    cbn [no_call_returns_to] in Hn.
    apply andb_true_iff in Hn as [Hnh Hnb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite forallb_forall in Hnh, Hkh.
    rewrite app_assoc. apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hin.
      exact (IHh k h Hin _ (Hnh (k, h) Hin) (Hkh (k, h) Hin)).
    + exact (IH _ Hnb Hkb).
  - intros ap ks Hn Hk. cbn [calls_kinded] in Hk |- *.
    cbn [no_call_returns_to] in Hn.
    apply negb_true_iff, orb_false_iff in Hn as [Hn _].
    rewrite lookup_app. destruct (lookup (ap_continuation ap) ks);
      [ exact Hk | ].
    cbn [lookup]. rewrite Hn. reflexivity.
  - intros params metadata handler IH ks Hn Hk. exact (IH ks Hn Hk).
Qed.

Lemma raises_kept_unused (r : continuation) (b : bool) :
  (∀ e : expr,
     ∀ xs : finite_map bool,
       no_call_returns_to r e = true -> raises_kept xs e = true ->
       raises_kept (xs ++ [(r, b)]) e = true)
  ∧ (∀ h : continuation_handler,
       ∀ xs : finite_map bool,
         handler_no_call_returns_to r h = true ->
         handler_raises_kept xs h = true ->
         handler_raises_kept (xs ++ [(r, b)]) h = true).
Proof.
  apply expr_handler_ind; try (intros; reflexivity).
  - intros x kind d body IH xs Hn Hk. exact (IH xs Hn Hk).
  - intros bindings body _ IH xs Hn Hk. cbn [raises_kept] in Hk |- *.
    apply andb_true_iff in Hk as [Hb Hk]. rewrite Hb.
    exact (IH xs Hn Hk).
  - intros k h body IHh IH xs Hn Hk. cbn [raises_kept] in Hk |- *.
    cbn [no_call_returns_to] in Hn.
    apply andb_true_iff in Hn as [Hnh Hnb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite (IHh xs Hnh Hkh).
    exact (IH ((k, handler_is_exn_handler h) :: xs) Hnb Hkb).
  - intros handlers body IHh IH xs Hn Hk. cbn [raises_kept] in Hk |- *.
    cbn [no_call_returns_to] in Hn.
    apply andb_true_iff in Hn as [Hnh Hnb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite forallb_forall in Hnh, Hkh.
    rewrite app_assoc. apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hin.
      exact (IHh k h Hin _ (Hnh (k, h) Hin) (Hkh (k, h) Hin)).
    + exact (IH _ Hnb Hkb).
  - intros ap xs Hn Hk. cbn [raises_kept] in Hk |- *.
    cbn [no_call_returns_to] in Hn.
    apply negb_true_iff, orb_false_iff in Hn as [_ Hn].
    rewrite lookup_app. destruct (lookup (ap_exn_continuation ap) xs);
      [ exact Hk | ].
    cbn [lookup]. rewrite Hn. exact Hk.
  - intros params metadata handler IH xs Hn Hk. exact (IH xs Hn Hk).
Qed.

Lemma lookup_rec_false (handlers : list (continuation * continuation_handler))
    (xs : finite_map bool) (k : continuation) :
  lookup k (map (fun '(k, _) => (k, false)) handlers ++ xs)
  = match lookup k handlers with
    | Some _ => Some false
    | None => lookup k xs
    end.
Proof.
  rewrite lookup_app.
  induction handlers as [ | [k' h'] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* No call raises to [r], where no [Let_cont] binds it again. *)
Fixpoint no_raise_to (r : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => no_raise_to r body
  | Let_cont_non_recursive k h body =>
    handler_no_raise_to r h && (Pos.eqb k r || no_raise_to r body)
  | Let_cont_recursive handlers body =>
    among r (map fst handlers)
    || (forallb (fun '(_, h) => handler_no_raise_to r h) handlers
        && no_raise_to r body)
  | Apply ap => negb (Pos.eqb (ap_exn_continuation ap) r)
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_no_raise_to (r : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => no_raise_to r handler
  end.

Lemma forallb_ext_in {A : Type} (f g : A -> bool) (l : list A) :
  (∀ x : A, x ∈ l -> f x = g x) -> forallb f l = forallb g l.
Proof.
  induction l as [ | x l IH]; intros H; [ reflexivity | ].
  cbn [forallb]. rewrite (H x (or_introl eq_refl)).
  rewrite IH; [ reflexivity | ]. intros y Hy. exact (H y (or_intror Hy)).
Qed.

Lemma raises_kept_ext :
  (∀ (e : expr) (xs xs' : finite_map bool),
     (∀ k : continuation, lookup k xs = lookup k xs') ->
     raises_kept xs e = raises_kept xs' e)
  ∧ (∀ (h : continuation_handler) (xs xs' : finite_map bool),
       (∀ k : continuation, lookup k xs = lookup k xs') ->
       handler_raises_kept xs h = handler_raises_kept xs' h).
Proof.
  apply expr_handler_ind; try (intros; reflexivity).
  - intros x kind d body IH xs xs' H. exact (IH xs xs' H).
  - intros bindings body _ IH xs xs' H. cbn [raises_kept].
    rewrite (IH xs xs' H). reflexivity.
  - intros k h body IHh IH xs xs' H. cbn [raises_kept].
    rewrite (IHh xs xs' H). f_equal. apply IH.
    intros j. cbn [lookup]. destruct (Pos.eqb j k); [ reflexivity | apply H ].
  - intros handlers body IHh IH xs xs' H. cbn [raises_kept].
    assert (H' : ∀ j : continuation,
               lookup j (map (fun '(k, _) => (k, false)) handlers ++ xs)
               = lookup j (map (fun '(k, _) => (k, false)) handlers ++ xs')).
    { intros j. rewrite !lookup_app.
      destruct (lookup j (map (fun '(k, _) => (k, false)) handlers));
        [ reflexivity | apply H ]. }
    rewrite (IH _ _ H'). f_equal. apply forallb_ext_in.
    intros [k h] Hin. exact (IHh k h Hin _ _ H').
  - intros ap xs xs' H. cbn [raises_kept].
    rewrite (H (ap_exn_continuation ap)). reflexivity.
  - intros params metadata handler IH xs xs' H. exact (IH xs xs' H).
Qed.

Lemma raises_kept_unraised (r : continuation) (b : bool) :
  (∀ e : expr,
     ∀ xs : finite_map bool,
       no_raise_to r e = true -> raises_kept xs e = true ->
       raises_kept (xs ++ [(r, b)]) e = true)
  ∧ (∀ h : continuation_handler,
       ∀ xs : finite_map bool,
         handler_no_raise_to r h = true ->
         handler_raises_kept xs h = true ->
         handler_raises_kept (xs ++ [(r, b)]) h = true).
Proof.
  assert (Hsh : ∀ xs : finite_map bool,
            lookup r xs <> None ->
            ∀ k : continuation, lookup k xs = lookup k (xs ++ [(r, b)])).
  { intros xs Hr k. rewrite lookup_app.
    destruct (lookup k xs) eqn:Hk; [ reflexivity | ].
    cbn [lookup]. destruct (Pos.eqb_spec k r) as [-> | _];
      [ contradiction | reflexivity ]. }
  apply expr_handler_ind; try (intros; reflexivity).
  - intros x kind d body IH xs Hn Hk. exact (IH xs Hn Hk).
  - intros bindings body _ IH xs Hn Hk. cbn [raises_kept] in Hk |- *.
    apply andb_true_iff in Hk as [Hb Hk]. rewrite Hb.
    exact (IH xs Hn Hk).
  - intros k h body IHh IH xs Hn Hk. cbn [raises_kept] in Hk |- *.
    cbn [no_raise_to] in Hn.
    apply andb_true_iff in Hn as [Hnh Hnb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite (IHh xs Hnh Hkh). cbn [andb].
    destruct (Pos.eqb_spec k r) as [-> | Hne].
    + rewrite <- (proj1 raises_kept_ext body
                    ((r, handler_is_exn_handler h) :: xs)); [ exact Hkb | ].
      apply (Hsh ((r, handler_is_exn_handler h) :: xs)).
      cbn [lookup]. rewrite Pos.eqb_refl. discriminate.
    + try rewrite (proj2 (Pos.eqb_neq k r) Hne) in Hnb. cbn [orb] in Hnb.
      exact (IH ((k, handler_is_exn_handler h) :: xs) Hnb Hkb).
  - intros handlers body IHh IH xs Hn Hk. cbn [raises_kept] in Hk |- *.
    cbn [no_raise_to] in Hn.
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite forallb_forall in Hkh.
    rewrite app_assoc.
    destruct (among r (map fst handlers)) eqn:Hin.
    + assert (Hl : lookup r (map (fun '(k, _) => (k, false)) handlers ++ xs)
                   <> None).
      { rewrite lookup_rec_false.
        destruct (lookup r handlers) as [hr | ] eqn:Hlr; [ discriminate | ].
        exfalso. apply among_In in Hin.
        exact (lookup_none_not_in r handlers Hlr Hin). }
      pose proof (Hsh _ Hl) as Hext.
      apply andb_true_intro. split.
      * apply forallb_forall. intros [k h] Hkh'.
        exact (eq_trans (eq_sym (proj2 raises_kept_ext h _ _ Hext))
                 (Hkh (k, h) Hkh')).
      * exact (eq_trans (eq_sym (proj1 raises_kept_ext body _ _ Hext)) Hkb).
    + cbn [orb] in Hn. apply andb_true_iff in Hn as [Hnh Hnb].
      rewrite forallb_forall in Hnh.
      apply andb_true_intro. split.
      * apply forallb_forall. intros [k h] Hkh'.
        exact (IHh k h Hkh' _ (Hnh (k, h) Hkh') (Hkh (k, h) Hkh')).
      * exact (IH _ Hnb Hkb).
  - intros ap xs Hn Hk. cbn [raises_kept] in Hk |- *.
    cbn [no_raise_to] in Hn. apply negb_true_iff in Hn.
    rewrite lookup_app. destruct (lookup (ap_exn_continuation ap) xs);
      [ exact Hk | ].
    cbn [lookup]. rewrite Hn. exact Hk.
  - intros params metadata handler IH xs Hn Hk. exact (IH xs Hn Hk).
Qed.

Lemma no_raise_of_raises_to (r : continuation) :
  (∀ e : expr, raises_to r e = false -> no_raise_to r e = true)
  ∧ (∀ h : continuation_handler,
       handler_raises_to r h = false -> handler_no_raise_to r h = true).
Proof.
  apply expr_handler_ind; try (intros; reflexivity).
  - intros x kind d body IH H. exact (IH H).
  - intros bindings body _ IH H. exact (IH H).
  - intros k h body IHh IH H. cbn [raises_to no_raise_to] in H |- *.
    apply orb_false_iff in H as [H1 H2].
    rewrite (IHh H1), (IH H2), orb_true_r. reflexivity.
  - intros handlers body IHh IH H. cbn [raises_to no_raise_to] in H |- *.
    apply orb_false_iff in H as [H1 H2].
    rewrite (IH H2), andb_true_r.
    destruct (among r (map fst handlers)); [ reflexivity | ].
    cbn [orb]. apply forallb_forall. intros [k h] Hin.
    apply (IHh k h Hin).
    destruct (handler_raises_to r h) eqn:E; [ | reflexivity ].
    rewrite (proj2 (existsb_exists _ _) (ex_intro _ (k, h) (conj Hin E)))
      in H1.
    discriminate H1.
  - intros ap H. cbn [raises_to no_raise_to] in H |- *. rewrite Pos.eqb_sym, H.
    reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Lemma free_of_raise (r : continuation) :
  (∀ e : expr, no_raise_to r e = false -> r ∈ free_continuations e)
  ∧ (∀ h : continuation_handler,
       handler_no_raise_to r h = false ->
       r ∈ handler_free_continuations h).
Proof.
  apply expr_handler_ind; try (intros; discriminate).
  - intros x kind d body IH H. exact (IH H).
  - intros bindings body _ IH H. exact (IH H).
  - intros k h body IHh IH H. cbn [no_raise_to] in H.
    cbn [free_continuations]. apply in_or_app.
    apply andb_false_iff in H as [H | H]; [ left; exact (IHh H) | right ].
    apply orb_false_iff in H as [Hk H].
    unfold without. apply filter_In. split; [ exact (IH H) | ].
    rewrite among_cons, Pos.eqb_sym, Hk. reflexivity.
  - intros handlers body IHh IH H. cbn [no_raise_to] in H.
    apply orb_false_iff in H as [Hin H].
    cbn [free_continuations]. unfold without. apply filter_In.
    split; [ | rewrite Hin; reflexivity ].
    apply in_or_app.
    apply andb_false_iff in H as [H | H]; [ left | right; exact (IH H) ].
    destruct (forallb (fun '(_, h) => handler_no_raise_to r h) handlers)
      eqn:E; [ discriminate H | ].
    apply not_true_iff_false in E.
    destruct (existsb (fun '(_, h) => negb (handler_no_raise_to r h))
                handlers) eqn:E'.
    + apply existsb_exists in E' as ([k h] & Hkh & Hn).
      apply negb_true_iff in Hn. apply in_flat_map.
      exists (k, h). split; [ exact Hkh | exact (IHh k h Hkh Hn) ].
    + exfalso. apply E. apply forallb_forall. intros [k h] Hkh.
      destruct (handler_no_raise_to r h) eqn:Hn; [ reflexivity | ].
      rewrite (proj2 (existsb_exists _ _)
                 (ex_intro _ (k, h) (conj Hkh (eq_trans
                    (f_equal negb Hn) eq_refl)))) in E'.
      discriminate E'.
  - intros ap H. cbn [no_raise_to] in H. apply negb_false_iff in H.
    apply Pos.eqb_eq in H. cbn [free_continuations]. right. left.
    exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* A primitive whose value may be of kind [Region] or [Rec_info] is the
   identity, or a primitive of regions, which has no value here. *)
Lemma region_prim (p : prim) :
  prim_result_kind p = Region ∨ prim_result_kind p = Rec_info ->
  (∃ k : flambda_kind, p = Unary (Opaque_identity k))
  ∨ ∃ r : region_primitive, p = Region_primitive r.
Proof.
  intros H.
  destruct p as [up | bp | vp | hp | r];
    [ destruct up | destruct bp | destruct vp | destruct hp
    | right; exists r; reflexivity ];
    cbn [prim_result_kind result_kind_of_unary_primitive
         result_kind_of_binary_primitive result_kind_of_variadic_primitive
         result_kind_of_heap_primitive] in H;
    try (left; eexists; reflexivity);
    unfold naked_immediate_kind, kind_of_standard_int, unboxed_kind in H;
    repeat (match type of H with
            | context [match ?x with _ => _ end] => destruct x
            end);
    destruct H as [H | H]; discriminate H.
Qed.

(* A value of another kind than [Value] is a constant. *)
Lemma not_value_const (v : value) :
  kind_of_value v <> Value -> ∃ c : const, v = Value_const c.
Proof.
  destruct v; simpl; intros H; try (destruct (H eq_refl)).
  eexists. reflexivity.
Qed.

Lemma region_not_value (k : flambda_kind) :
  k = Region ∨ k = Rec_info -> k <> Value.
Proof. intros [-> | ->]; discriminate. Qed.

(* The [Let]s and the non-recursive [Let_cont]s that begin an expression:
   what the rebuild may remove without a step. *)
Fixpoint spine (e : expr) : nat :=
  match e with
  | Let _ _ _ body | Let_cont_non_recursive _ _ body => S (spine body)
  | _ => 0
  end.

Definition measure (s : state) : nat := spine (state_expr s).

(* ======================================================================== *)
(* The rebuild with no change of calling convention                         *)
(* ======================================================================== *)

(* The rebuild with changes of calling convention
   (FlambdaReaperCallingConventions.v) is [rebuild_expr], case for case,
   where no code of the unit changes: the runs below are of the first,
   and so of the second. *)
Section Unchanged.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable first_fresh : BinNums.positive.

(* Every code of the unit is in the table. *)
Hypothesis Hall : ∀ id : code_id,
  among id unit_code_ids = true -> cannot_change (Node_code_id id) = true.

Lemma code_calling_convention_change_unchanged : ∀ id metadata,
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    metadata = Not_changing_calling_convention.
Proof.
  intros id metadata. unfold code_calling_convention_change,
    cannot_change_calling_convention.
  destruct (among id unit_code_ids) eqn:Hamong; [ | reflexivity ].
  rewrite Hall by exact Hamong. reflexivity.
Qed.

(* A call changes no calling convention. *)
Lemma rebuild_apply_cc_unchanged : ∀ en ap,
  rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes first_fresh en ap
  = rebuild_apply F sol preserve_direct_calls unit_code_ids first_fresh en ap.
Proof.
  intros en ap. unfold rebuild_apply_cc, apply_calling_convention_change.
  destruct (code_id_actually_called F sol ap) as [id | ]; [ | reflexivity ].
  unfold get_calling_convention_change.
  destruct (lookup id codes) as [metadata | ]; [ | reflexivity ].
  rewrite code_calling_convention_change_unchanged. reflexivity.
Qed.

(* The two rebuilds are the same, case for case. *)
Fixpoint rebuild_expr_cc_unchanged (en : rebuild_env) (e : expr) {struct e} :
  rebuild_expr_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes first_fresh en e
  = rebuild_expr F sol preserve_direct_calls unit_code_ids first_fresh en e
with rebuild_handler_cc_unchanged (en : rebuild_env) (dead_check : bool)
    (mask : list bool) (h : continuation_handler) {struct h} :
  rebuild_handler_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes first_fresh en dead_check mask h
  = rebuild_handler F sol preserve_direct_calls unit_code_ids first_fresh en
      dead_check mask h
with rebuild_binding_cc_unchanged (en : rebuild_env)
    (binding : static_binding) {struct binding} :
  rebuild_binding_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes first_fresh en binding
  = rebuild_binding F sol preserve_direct_calls unit_code_ids first_fresh en
      binding
with rebuild_code_cc_unchanged (id : code_id) (c : code) {struct c} :
  rebuild_code_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes first_fresh id c
  = rebuild_code F sol preserve_direct_calls unit_code_ids first_fresh c.
Proof.
  - destruct e as [x kind defining_expr body | bindings body | k h body
                  | handlers body | ap | ac | scrutinee arms | ].
    + simpl. destruct (is_pending_projection en x defining_expr).
      * rewrite rebuild_expr_cc_unchanged. reflexivity.
      * destruct defining_expr; rewrite ?rebuild_expr_cc_unchanged;
          reflexivity.
    + simpl. rewrite rebuild_expr_cc_unchanged. f_equal.
      induction bindings as [ | binding bindings IH ]; [ reflexivity | ].
      simpl. rewrite rebuild_binding_cc_unchanged, IH.
      reflexivity.
    + simpl. rewrite rebuild_expr_cc_unchanged.
      rewrite rebuild_handler_cc_unchanged. reflexivity.
    + simpl.
      set (en' := bind_continuations en _).
      rewrite rebuild_expr_cc_unchanged. f_equal. clearbody en'.
      induction handlers as [ | [k h] handlers IH ]; [ reflexivity | ].
      simpl. rewrite rebuild_handler_cc_unchanged, IH. reflexivity.
    + apply rebuild_apply_cc_unchanged.
    + reflexivity.
    + reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite rebuild_expr_cc_unchanged. reflexivity.
  - destruct binding as [id c | id | s const]; simpl.
    + rewrite rebuild_code_cc_unchanged. reflexivity.
    + reflexivity.
    + reflexivity.
  - destruct c as [params my_closure return_continuation exn_continuation
                   metadata body].
    simpl. rewrite code_calling_convention_change_unchanged.
    rewrite rebuild_expr_cc_unchanged. reflexivity.
Qed.

End Unchanged.

(* ======================================================================== *)
(* A call that changes its calling convention                               *)
(* ======================================================================== *)

(* What [rebuild_apply_cc] makes of the arguments of a call, and of its
   arities, in terms of the masks of FlambdaReaperRebuild.v: the
   arguments kept are those that [filter_kept] keeps at the decisions of
   the callee, and the arities are those of the rebuilt code
   ([changed_code_metadata]). *)

Lemma group_by_sizes_concat {A : Type} (sizes : list nat) (l : list A) :
  length l = list_sum sizes -> concat (group_by_sizes sizes l) = l.
Proof.
  revert l. induction sizes as [ | n sizes IH]; intros l Hl.
  - destruct l; [ reflexivity | discriminate Hl ].
  - cbn [group_by_sizes concat]. rewrite IH; [ apply firstn_skipn | ].
    rewrite length_skipn, Hl. simpl. lia.
Qed.

Lemma filter_kept_app {A : Type} (m1 m2 : list bool) (l1 l2 : list A) :
  length m1 = length l1 ->
  filter_kept (m1 ++ m2) (l1 ++ l2) = filter_kept m1 l1 ++ filter_kept m2 l2.
Proof.
  revert m1. induction l1 as [ | a l1 IH]; intros [ | b m1] Hl;
    try discriminate Hl; [ reflexivity | ].
  injection Hl as Hl. cbn [app filter_kept].
  destruct b; [ rewrite (IH m1 Hl); reflexivity | exact (IH m1 Hl) ].
Qed.

(* The mask of [apply_decisions], for [n] results. *)
Definition pad_mask (n : nat) (mask : list bool) : list bool :=
  firstn n (mask ++ repeat true n).

Lemma length_pad_mask (n : nat) (mask : list bool) :
  length (pad_mask n mask) = n.
Proof.
  unfold pad_mask. rewrite length_firstn, length_app, repeat_length. lia.
Qed.

Lemma pad_mask_nil (n : nat) : pad_mask n [] = repeat true n.
Proof.
  unfold pad_mask. cbn [app]. apply firstn_all2. rewrite repeat_length.
  lia.
Qed.

Lemma pad_mask_self (m : list bool) : pad_mask (length m) m = m.
Proof.
  unfold pad_mask. rewrite firstn_app, Nat.sub_diag, firstn_all.
  cbn [firstn]. apply app_nil_r.
Qed.


Lemma forallb_repeat_true (n : nat) :
  forallb (fun keep => keep) (repeat true n) = true.
Proof. induction n as [ | n IH]; [ reflexivity | exact IH ]. Qed.

Lemma filter_kept_pad_gen {A : Type} (l : list A) :
  ∀ (mask : list bool) (m : nat),
    length l <= m ->
    filter_kept (firstn (length l) (mask ++ repeat true m)) l
    = filter_kept mask l.
Proof.
  induction l as [ | a l IH]; intros mask m Hm; [ reflexivity | ].
  destruct m as [ | m]; [ simpl in Hm; lia | ].
  destruct mask as [ | b mask].
  - cbn [app length repeat firstn filter_kept]. f_equal.
    exact (IH [] m ltac:(simpl in Hm; lia)).
  - cbn [app length firstn filter_kept].
    rewrite (IH mask (S m)) by (simpl in Hm; lia). reflexivity.
Qed.

Lemma filter_kept_pad {A : Type} (mask : list bool) (l : list A) :
  filter_kept (pad_mask (length l) mask) l = filter_kept mask l.
Proof. exact (filter_kept_pad_gen l mask (length l) (le_n _)). Qed.

Lemma kept_kinds_app (d1 d2 : list param_decision) :
  kept_kinds (d1 ++ d2) = kept_kinds d1 ++ kept_kinds d2.
Proof. unfold kept_kinds. apply flat_map_app. Qed.

Section Model.

Variable sol : solution.

(* The decision at [i]. *)
Lemma decisions_of_nth (nodes : list node) (kinds : list flambda_kind)
    (i : nat) (d : param_decision) :
  nth_error (decisions_of sol nodes kinds) i = Some d ->
  ∃ (n : node) (kind : flambda_kind),
    nth_error nodes i = Some n ∧ nth_error kinds i = Some kind
    ∧ d = if is_node_used sol n kind then Keep kind else Delete.
Proof.
  unfold decisions_of. rewrite nth_error_map. intros Hd.
  destruct (nth_error (combine nodes kinds) i) as [[n kind] | ] eqn:Hc;
    [ | discriminate Hd ].
  injection Hd as <-. exists n, kind.
  destruct (nth_error_combine_inv _ _ _ _ _ Hc) as [Hn Hk].
  auto.
Qed.

Lemma length_decisions_of (nodes : list node) (kinds : list flambda_kind) :
  length (decisions_of sol nodes kinds)
  = Nat.min (length nodes) (length kinds).
Proof. unfold decisions_of. rewrite length_map. apply length_combine. Qed.

(* Each decision keeps a value of the kind it was taken at. *)
Lemma kept_kinds_decisions_of (nodes : list node) (kinds : list flambda_kind) :
  length nodes = length kinds ->
  kept_kinds (decisions_of sol nodes kinds)
  = filter_kept (map is_kept (decisions_of sol nodes kinds)) kinds.
Proof.
  revert kinds. induction nodes as [ | n nodes IH];
    intros [ | kind kinds] Hl; try discriminate Hl; [ reflexivity | ].
  injection Hl as Hl. unfold decisions_of in *. cbn [combine map].
  destruct (is_node_used sol n kind);
    cbn [kept_kinds flat_map is_kept map filter_kept app];
    unfold kept_kinds in IH; rewrite (IH kinds Hl); reflexivity.
Qed.

Lemma get_args_with_kinds_fst (en : rebuild_env)
    (ds : list param_decision) (args : list simple) :
  length ds = length args ->
  map fst (get_args_with_kinds sol en ds args)
  = filter_kept (map is_kept ds) (map (rewrite_simple sol en) args).
Proof.
  unfold get_args_with_kinds. revert ds.
  induction args as [ | a args IH]; intros [ | d ds] Hl;
    try discriminate Hl; [ reflexivity | ].
  injection Hl as Hl. cbn [combine flat_map].
  destruct d as [kind | ]; cbn [app map fst is_kept filter_kept];
    rewrite (IH ds Hl); reflexivity.
Qed.

Lemma get_args_with_kinds_snd (en : rebuild_env)
    (ds : list param_decision) (args : list simple) :
  length ds = length args ->
  map snd (get_args_with_kinds sol en ds args) = kept_kinds ds.
Proof.
  unfold get_args_with_kinds. revert ds.
  induction args as [ | a args IH]; intros [ | d ds] Hl;
    try discriminate Hl; [ reflexivity | ].
  injection Hl as Hl. cbn [combine flat_map].
  destruct d as [kind | ]; cbn [app map snd kept_kinds flat_map];
    unfold kept_kinds in IH; rewrite (IH ds Hl); reflexivity.
Qed.

(* The groups of the arguments of a call, at the decisions of the
   callee. *)
Definition cc_groups (en : rebuild_env) (sizes : list nat)
    (args : list simple) (ds : list param_decision)
    : list (list (simple * flambda_kind)) :=
  map (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
    (combine (group_by_sizes sizes args) (group_by_sizes sizes ds)).

(* The arguments kept. *)
Lemma cc_groups_args (en : rebuild_env) (sizes : list nat)
    (args : list simple) (ds : list param_decision) :
  length args = list_sum sizes -> length ds = list_sum sizes ->
  flat_map (map fst) (cc_groups en sizes args ds)
  = filter_kept (map is_kept ds) (map (rewrite_simple sol en) args).
Proof.
  unfold cc_groups. revert args ds.
  induction sizes as [ | n sizes IH]; intros args ds Ha Hd.
  - destruct args; [ | discriminate Ha ].
    destruct ds; [ reflexivity | discriminate Hd ].
  - simpl in Ha, Hd. cbn [group_by_sizes combine map flat_map].
    rewrite (IH (skipn n args) (skipn n ds))
      by (rewrite length_skipn; lia).
    rewrite get_args_with_kinds_fst
      by (rewrite !length_firstn; lia).
    transitivity
      (filter_kept (map is_kept (firstn n ds ++ skipn n ds))
         (map (rewrite_simple sol en) (firstn n args ++ skipn n args)));
      [ | rewrite !firstn_skipn; reflexivity ].
    rewrite !map_app, filter_kept_app; [ reflexivity | ].
    rewrite !length_map, !length_firstn. lia.
Qed.

(* Their arity, one product to a group, is that of the rebuilt code. *)
Lemma cc_groups_arity (en : rebuild_env) (sizes : list nat)
    (args : list simple) (ds : list param_decision) :
  length args = list_sum sizes -> length ds = list_sum sizes ->
  arity_of_groups (cc_groups en sizes args ds)
  = arity_of_grouped_decisions (group_by_sizes sizes ds).
Proof.
  unfold cc_groups, arity_of_groups, arity_of_grouped_decisions.
  revert args ds.
  induction sizes as [ | n sizes IH]; intros args ds Ha Hd;
    [ reflexivity | ].
  simpl in Ha, Hd. cbn [group_by_sizes combine map].
  rewrite (IH (skipn n args) (skipn n ds)) by (rewrite length_skipn; lia).
  f_equal. f_equal.
  rewrite <- (get_args_with_kinds_snd en (firstn n ds) (firstn n args))
    by (rewrite !length_firstn; lia).
  rewrite map_map. apply map_ext. intros [s kind]. reflexivity.
Qed.

Lemma length_cc_groups (en : rebuild_env) (sizes : list nat)
    (args : list simple) (ds : list param_decision) :
  length (cc_groups en sizes args ds) = length sizes.
Proof.
  unfold cc_groups. rewrite length_map, length_combine,
    !length_group_by_sizes.
  apply Nat.min_id.
Qed.

End Model.

(* The arity of the products of kept kinds, unarized. *)
Lemma unarize_arity_of_grouped_decisions (groups : list (list param_decision))
    :
  unarize_per_parameter (arity_of_grouped_decisions groups)
  = map kept_kinds groups.
Proof.
  unfold unarize_per_parameter, arity_of_grouped_decisions.
  rewrite map_map. apply map_ext. intros ds. cbn [unarize_component].
  induction (kept_kinds ds) as [ | k ks IH]; [ reflexivity | ].
  cbn [map flat_map unarize_component app]. rewrite IH. reflexivity.
Qed.

Lemma unarize_arity_of_decisions (ds : list param_decision) :
  unarize (arity_of_decisions ds) = kept_kinds ds.
Proof.
  unfold unarize, arity_of_decisions. cbn [flat_map unarize_component].
  rewrite app_nil_r.
  induction (kept_kinds ds) as [ | k ks IH]; [ reflexivity | ].
  cbn [map flat_map unarize_component app]. rewrite IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

(* The witness of unknown arity of the tupled code [id], whose one
   parameter is the tuple, the parameters of the code being read off its
   fields, and the two facts that keep the tuple alive, a read of
   [Is_int] of it and a use of that by the code id
   ([create_unknown_arity_tupled_call_witnesses]). *)
Definition tupled_kept (G : graph) (id : code_id) (cd : code_dep) : Prop :=
  ∃ W U K : node,
    cd_unknown_arity_call_witnesses cd = [W]
    ∧ g_parameter G W 0 U
    ∧ (∀ (i : nat) (p : node),
         nth_error (cd_params cd) i = Some p ->
         g_accessor G p (Field_block i Value) U)
    ∧ g_accessor G K Field_is_int U
    ∧ g_use G (Node_code_id id) K.

Section Run.

(* The setting of FlambdaReaperFlowProofs.flow_invariant, and the
   arguments of the rebuild. *)
Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable u : unit_interface.
Variable F : facts.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable first_fresh : positive.
Variable exact : bool.
(* The table of [cannot_change_calling_convention], the code deps and the
   metadata of the code of the unit, of the rebuild with changes of
   calling convention (FlambdaReaperCallingConventions.v). *)
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable code_metadatas : finite_map code_metadata.

Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hexact : field_used_exact is_local G sol.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hsyms :
  ∀ s : symbol, noted s -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hunit_syms :
  ∀ s : symbol,
    among s symbols = true -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hdeps : ∀ d : closure_dep, deps d -> closure_dep_facts G cds d.
Hypothesis Hnd : NoDup (map fst cds).
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep),
    lookup id cds = Some cd -> known_witness_edges G id cd.
Hypothesis HG : G = graph_of_facts F.
Hypothesis Hunit :
  ∀ (k : continuation) (p : node), p ∈ unit_cont_nodes u k -> g_any_usage G p.
(* The code of the unit is as the unit asks: what the body of code is run
   under. *)
Hypothesis Hcode_checks :
  ∀ (id : code_id) (c : code),
    (id, c) ∈ codes ->
    scrutinees_used G (Some id) (code_body c)
    ∧ checked exact codes (code_body c) = true
    ∧ raises_kept [] (code_body c) = true
    ∧ code_is_closed c = true
    ∧ calls_kinded [] (code_body c) = true
    ∧ among first_fresh (continuations_of (code_body c)) = false.
(* The tuple that a call gives tupled code is kept alive
   (FlambdaReaperTupledSpec.v). *)
Hypothesis Hkeep_alive :
  ∀ (id : code_id) (c : code) (cd : code_dep),
    (id, c) ∈ codes -> code_is_tupled c = true -> lookup id cds = Some cd ->
    tupled_kept G id cd.
(* The table is closed under the rules of
   [cannot_change_calling_convention], with a table of the witnesses
   (FlambdaReaperCallingConventionsSpec.cc_closed). *)
Variable cannot_change_witness : node -> bool.
Hypothesis Hcc_closed : cc_closed G sol cannot_change_witness cannot_change.
(* Where code changes its calling convention: the witnesses of known
   arity that calls read off a node have the same sources
   (FlambdaReaperCallingConventionsSpec.check_apply_widgets), and a
   witness gives one code id. *)
Hypothesis Hwidgets :
  ∀ (id : code_id) (a a' n y : node),
    cannot_change (Node_code_id id) = false ->
    g_accessor G a (Field_call_witness Known_arity_code_pointer) n ->
    g_accessor G a' (Field_call_witness Known_arity_code_pointer) n ->
    (s_any_source sol a' = true -> s_any_source sol a = true)
    ∧ (s_sources sol a' y = true -> s_sources sol a y = true).
(* The code of the unit, where it may be called: no call raises to its
   return continuation, and what returns there is as
   FlambdaReaperCallingConventionsSpec.ret_calls_ok says (from
   [well_formed_unit] and [results_kinded]). *)
Hypothesis Hcode_rets :
  ∀ (id : code_id) (params : list variable) (mc : variable)
    (ret exn : continuation) (md : code_metadata) (body : expr),
    exact = false ->
    (id, Code params mc ret exn md body) ∈ codes ->
    (ret <> exn -> no_raise_to ret body = true)
    ∧ (length params = length (unarize (cm_params_arity md)) ->
       ret_calls_ok
         (rev (with_kinds params (unarize (cm_params_arity md)))
          ++ [(mc, Value)])
         ret (unarize (cm_result_arity md)) body
       = true).
(* Where the code [id] changes its calling convention: its code dep is
   that of the traversal, and its metadata that of its code. *)
Hypothesis Hcode_deps :
  ∀ id : code_id,
    cannot_change (Node_code_id id) = false ->
    lookup id code_deps = lookup id cds.
Hypothesis Hmetadatas :
  ∀ (id : code_id) (c : code),
    (id, c) ∈ codes -> cannot_change (Node_code_id id) = false ->
    lookup id code_metadatas = Some (metadata_of_code c).
Hypothesis Hcode_id_of_witness :
  ∀ (id id' : code_id) (w : node),
    cannot_change (Node_code_id id) = false ->
    g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
    g_constructor G w Field_code_id_of_call_witness (Node_code_id id') ->
    id' = id.

Abbreviation slots := (∀ f : field, is_local f = false).
Abbreviation called := (called_by G cds).
(* The first run of the statements here makes no cell, and has no
   location. *)
Abbreviation no_loc := (fun _ : location => @None node).
Abbreviation flows_ := (flows G sol keep called no_loc).
Abbreviation env_flows_ := (env_flows G sol keep called no_loc).
Abbreviation statics_flow_ := (statics_flow G sol keep called no_loc).
Abbreviation edges :=
  (expr_edges G noted deps cds keep codes symbols slots False).
Abbreviation handler_edges_ :=
  (handler_edges G noted deps cds keep codes symbols slots False).
Abbreviation binding_edges_ :=
  (binding_edges G noted deps cds keep codes symbols slots False).
Abbreviation named_edges_ := (named_edges G noted deps keep slots False).
Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hs n := (s_has_source sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).
Abbreviation rel := (rel G sol keep called (exact = true) True).
Abbreviation field_rel := (field_rel G sol keep called (exact = true) True).
Abbreviation slot_rel := (slot_rel G sol keep called (exact = true) True).
Abbreviation sim := (sim sol (exact = true) True).
Abbreviation rebuild :=
  (rebuild_expr_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids code_metadatas first_fresh).
Abbreviation rebuild_h :=
  (rebuild_handler_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids code_metadatas first_fresh).
Abbreviation rebuild_b :=
  (rebuild_binding_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids code_metadatas first_fresh).
Abbreviation rebuild_c :=
  (rebuild_code_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids code_metadatas first_fresh).
Abbreviation rcalls_ok :=
  ret_calls_ok.

(* The code [id] does not change its calling convention. *)
Definition unchanged (id : code_id) (c : code) : Prop :=
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    (metadata_of_code c)
  = Not_changing_calling_convention.

(* A call of unknown arity is rebuilt as in FlambdaReaperRebuild.v. *)
Lemma rebuild_apply_indirect (en : rebuild_env) (ap : apply_expr) :
  ap_call_kind ap = Indirect ->
  rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids code_metadatas first_fresh en ap
  = rebuild_apply F sol preserve_direct_calls unit_code_ids first_fresh en ap.
Proof.
  intros Hk. unfold rebuild_apply_cc, apply_calling_convention_change,
    code_id_actually_called.
  rewrite Hk. reflexivity.
Qed.

(* The rebuild of a call: that of FlambdaReaperRebuild.v, [Invalid], or
   the wrapper of a call that changes its calling convention, which
   returns and raises where the call does. *)
Lemma rebuild_apply_cc_cases (en : rebuild_env) (ap : apply_expr) :
  rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids code_metadatas first_fresh en ap
  = rebuild_apply F sol preserve_direct_calls unit_code_ids first_fresh en ap
  ∨ rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
      unit_code_ids code_metadatas first_fresh en ap
    = Invalid
  ∨ ∃ (ap' : apply_expr) (fd : list (param_decision * flambda_kind)),
      rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
        unit_code_ids code_metadatas first_fresh en ap
      = make_apply_wrapper_cc first_fresh en ap' fd
      ∧ ap_continuation ap' = ap_continuation ap
      ∧ ap_exn_continuation ap' = ap_exn_continuation ap.
Proof.
  unfold rebuild_apply_cc.
  destruct (apply_calling_convention_change F sol cannot_change code_deps
              unit_code_ids code_metadatas ap)
    as [ | pd rd ]; [ left; reflexivity | right ].
  destruct (_ || _); [ left; reflexivity | right ].
  eexists _, _. split; [ reflexivity | split; reflexivity ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Environments                                                             *)
(* ------------------------------------------------------------------------ *)

(* A value of the kind [Region] or [Rec_info] is the poison of that
   kind. *)
Lemma region_value (v : value) :
  argument_kind_ok (kind_of_value v) = false ->
  v = Value_const (Poison (kind_of_value v)).
Proof.
  destruct v as [c | | | | | | | | ]; simpl; try discriminate.
  destruct c; simpl; try discriminate. reflexivity.
Qed.

(* An argument of the kind [Region] or [Rec_info], among arguments of the
   same kinds in the two runs, is the same in the two, and related at
   any node. *)
Lemma rel_region_argument (vs vs' : list value) (v v' : value) (i : nat)
    (n : node) :
  argument_kind_ok (kind_of_value v) = false ->
  map kind_of_value vs' = map kind_of_value vs ->
  nth_error vs i = Some v -> nth_error vs' i = Some v' -> rel v n v'.
Proof.
  intros Hk Hkinds Hv Hv'.
  assert (Hkv' : kind_of_value v' = kind_of_value v).
  { apply (f_equal (fun l => nth_error l i)) in Hkinds.
    rewrite !nth_error_map, Hv, Hv' in Hkinds. injection Hkinds as E.
    exact E. }
  rewrite (region_value v' ltac:(rewrite Hkv'; exact Hk)),
    (region_value v Hk), Hkv'.
  cbn [rel kind_of_const]. split; [ | reflexivity ].
  intros H. rewrite H in Hk. discriminate Hk.
Qed.

(* Every variable of the first run has its kind in [kinds], and, if the
   rebuild keeps it, is bound in the second to a related value. *)
Definition env_ok (kinds : finite_map flambda_kind) (vars vars' : env)
    : Prop :=
  ∀ (x : variable) (v : value),
    lookup x vars = Some v ->
    lookup x kinds = Some (kind_of_value v)
    ∧ (is_var_used sol x (kind_of_value v) = true ->
       ∃ v' : value, lookup x vars' = Some v' ∧ rel v (Node_variable x) v').

(* A simple that is a constant or has any usage: in the second run, a
   value [sim] to it, and related to it at a variable that has any source
   and uses the simple. *)
Definition element_rel (w w' : value) : Prop :=
  (w' = w
   ∧ ((∃ c : const, w = Value_const c)
      ∨ ∃ s : symbol, w = Value_symbol s ∧ au (Node_symbol s)))
  ∨ ∃ m : node, au m ∧ rel w m w'.

Lemma element_rel_at (w w' : value) (x : variable) :
  element_rel w w' -> as_ (Node_variable x) -> rel w (Node_variable x) w'.
Proof.
  intros [(-> & [(c & ->) | (s & -> & Hs)]) | (m & Hm & Hrel)] Hx.
  - cbn [rel]. split; [ intros _; exact Hx | reflexivity ].
  - cbn [rel]. split; [ reflexivity | split ].
    + exact (may_come_from_any is_local G sol Hc _ _ Hx (fun _ => Hs)).
    + intros _. exact (au_hu _ _ _ Hc _ Hs).
  - exact (rel_escape G sol keep called (exact = true) True is_local Hc w _ _
      w' Hx
             (fun _ => Hm) Hrel).
Qed.

Lemma element_rel_sim (w w' : value) : element_rel w w' -> sim w w'.
Proof.
  intros [(-> & [(c & ->) | (s & -> & Hs)]) | (m & Hm & Hrel)].
  - reflexivity.
  - split; [ reflexivity | exact Hs ].
  - exact (rel_au_sim G sol keep called (exact = true) True is_local Hc Hlocal
      _ _
    _ Hm
             Hrel).
Qed.

(* A boxed number, a string or an immutable array that the first run
   binds to [s], and what the second binds to [s] ([found]): nothing, or
   the same but for the contents of a boxed number, which are the same or
   [sim] (those of a variable with any usage), and the elements of an
   array, which are related ([element_rel]) if [s] has any usage; and
   something if [s] has a use. *)
Definition static_other_rel (s : symbol)
    (allocated : statically_allocated_value)
    (found : option statically_allocated_value) : Prop :=
  match allocated with
  | Static_boxed_number v =>
    found = None
    ∨ ∃ v' : value,
        found = Some (Static_boxed_number v') ∧ (v' = v ∨ sim v v')
  | Static_string _ =>
    found = None ∨ found = Some allocated
  | Static_immutable_array elements =>
    found = None
    ∨ ∃ elements' : list value,
        found = Some (Static_immutable_array elements')
        ∧ (au (Node_symbol s) -> Forall2 element_rel elements elements')
  | _ => False
  end
  ∧ (hu (Node_symbol s) -> found <> None).

(* What the second static environment binds, the first does; a symbol
   with a use is bound in both to related blocks; and the first binds
   only blocks and closures of static sets, which the second binds with
   related value slots or not at all, and binds if they have a use, and
   boxed numbers, strings and immutable arrays ([static_other_rel]); the
   second binds the code of the first, rebuilt, or none if the code id
   has no use. *)
Definition statics_rel (statics statics' : static_env) : Prop :=
  (∀ s : symbol,
     lookup s (static_values statics) = None ->
     lookup s (static_values statics') = None)
  ∧ (∀ (s : symbol) (tag : Z) (fields : list value),
       lookup s (static_values statics) = Some (Static_block tag fields) ->
       hu (Node_symbol s) ->
       ∃ fields' : list value,
         lookup s (static_values statics') = Some (Static_block tag fields')
         ∧ fields_related (field_rel (Node_symbol s)) 0 fields fields')
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
       lookup s (static_values statics) = Some allocated ->
       (∃ (tag : Z) (fields : list value),
          allocated = Static_block tag fields)
       ∨ (∃ (slot : function_slot)
            (closure_symbols : list (function_slot * symbol))
            (function_decls : list (function_slot * code_id))
            (vs : list (value_slot * value)),
            allocated
            = Static_closure_of_set slot closure_symbols function_decls vs
            ∧ (lookup s (static_values statics') = None
               ∨ ∃ vs' : list (value_slot * value),
                   lookup s (static_values statics')
                   = Some (Static_closure_of_set slot closure_symbols
                             function_decls vs')
                   ∧ (exact = true -> vs' = vs)
                   ∧ slots_related (slot_rel (Node_symbol s) vs') vs [])
            ∧ (hu (Node_symbol s) ->
               lookup s (static_values statics') <> None))
       ∨ static_other_rel s allocated (lookup s (static_values statics')))
  ∧ (∀ id : code_id,
       lookup id (static_code statics')
       = option_map (rebuild_c id) (lookup id (static_code statics))
       ∨ (lookup id (static_code statics') = None
          ∧ ~ hu (Node_code_id id))).

(* The simples that [rewrite_simple] keeps. *)
Definition kept_simple (en : rebuild_env) (s : simple) : bool :=
  match s with
  | Var x => is_var_used sol x (kind_of_variable en x)
  | Symbol sym => is_symbol_used sol sym
  | Const _ => true
  end.

Lemma is_var_used_hu (x : variable) (k : flambda_kind) :
  hu (Node_variable x) -> is_var_used sol x k = true.
Proof. intros H. destruct k; simpl; auto. Qed.

Lemma is_var_used_cases (x : variable) (k : flambda_kind) :
  is_var_used sol x k = true ->
  hu (Node_variable x) ∨ (k = Region ∨ k = Rec_info).
Proof. destruct k; simpl; auto. Qed.

Lemma kept_of_hu (en : rebuild_env) (s : simple) :
  hu (node_of_simple s) -> kept_simple en s = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros H; [ | exact H | reflexivity ].
  apply is_var_used_hu, H.
Qed.

Lemma env_ok_kind (en : rebuild_env) (vars vars' : env) (x : variable)
    (v : value) :
  env_ok (env_kinds en) vars vars' -> lookup x vars = Some v ->
  kind_of_variable en x = kind_of_value v.
Proof.
  intros Henv Hx. unfold kind_of_variable.
  rewrite (proj1 (Henv x v Hx)). reflexivity.
Qed.

(* A simple whose value is of kind [Region] or [Rec_info] is kept. *)
Lemma kept_of_region (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_ok (env_kinds en) vars vars' -> eval_simple vars s = Some v ->
  kind_of_value v = Region ∨ kind_of_value v = Rec_info ->
  kept_simple en s = true.
Proof.
  intros Henv Hs Hk. destruct s as [x | sym | c]; simpl in Hs |- *;
    [ | | reflexivity ].
  - rewrite (env_ok_kind en vars vars' x v Henv Hs).
    destruct Hk as [-> | ->]; reflexivity.
  - injection Hs as <-. simpl in Hk. destruct Hk; discriminate.
Qed.

(* A simple of the first run, and what the rebuild makes of it in the
   second: of the same kind, and related if it is kept. *)
Lemma rewrite_simple_eval (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_ok (env_kinds en) vars vars' ->
  simple_noted noted s ->
  eval_simple vars s = Some v ->
  ∃ v' : value,
    eval_simple vars' (rewrite_simple sol en s) = Some v'
    ∧ kind_of_value v' = kind_of_value v
    ∧ (kept_simple en s = true ->
       rewrite_simple sol en s = s ∧ rel v (node_of_simple s) v').
Proof.
  intros Henv Hnoted Hs.
  destruct s as [x | sym | c]; simpl in Hs, Hnoted.
  - destruct (Henv x v Hs) as [_ Hused].
    pose proof (env_ok_kind en vars vars' x v Henv Hs) as Hk.
    unfold rewrite_simple, kept_simple. cbv zeta. rewrite Hk.
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hu.
    + destruct (Hused eq_refl) as (v' & Hl & Hrel). exists v'.
      split;
        [ exact Hl | split; [ exact (rel_kind _ _ _ _ _ _ _ _ _ Hrel) | ] ].
      intros _. split; [ reflexivity | exact Hrel ].
    + exists (Value_const (Poison (kind_of_value v))).
      split; [ reflexivity | split; [ reflexivity | discriminate ] ].
  - injection Hs as <-. unfold rewrite_simple, kept_simple.
    destruct (is_symbol_used sol sym) eqn:Hu.
    + exists (Value_symbol sym).
      split; [ reflexivity | split; [ reflexivity | ] ].
      intros _. split; [ reflexivity | ].
      split; [ reflexivity | split; [ exact (Hsyms sym Hnoted) | auto ] ].
    + exists (Value_const (Poison Value)).
      split; [ reflexivity | split; [ reflexivity | discriminate ] ].
  - injection Hs as <-. exists (Value_const c).
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros _. split; [ reflexivity | ].
    split; [ intros _; exact (consts_any_source is_local G sol Hc Hconsts)
           | reflexivity ].
Qed.

Lemma rewrite_simples_eval (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (args : list simple) (vs : list value),
    simples_noted noted args ->
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (map (rewrite_simple sol en) args) = Some vs'
      ∧ map kind_of_value vs' = map kind_of_value vs
      ∧ ∀ (i : nat) (s : simple) (v v' : value),
          nth_error args i = Some s -> nth_error vs i = Some v ->
          nth_error vs' i = Some v' -> kept_simple en s = true ->
          rel v (node_of_simple s) v'.
Proof.
  intros Henv. induction args as [ | s args IH]; intros vs Hnoted Hvs;
    simpl in Hvs.
  - injection Hvs as <-. exists [].
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros [ | i]; discriminate.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-.
    destruct (rewrite_simple_eval en vars vars' s v Henv
                (Hnoted s (or_introl eq_refl)) Hs)
      as (v' & Hs' & Hk & Hkept).
    destruct (IH ws (fun s' Hin => Hnoted s' (or_intror Hin)) eq_refl)
      as (ws' & Hws' & Hks & Hrels).
    exists (v' :: ws'). simpl. rewrite Hs', Hws'.
    split; [ reflexivity | split; [ simpl; rewrite Hk, Hks; reflexivity | ] ].
    intros [ | i] s'' w w' Hi Hw Hw'; simpl in Hi, Hw, Hw'.
    + injection Hi as <-. injection Hw as <-. injection Hw' as <-.
      intros Hkeep. exact (proj2 (Hkept Hkeep)).
    + exact (Hrels i s'' w w' Hi Hw Hw').
Qed.

(* The simples rewritten are of the same kinds. *)
Lemma rewrite_simples_kinds (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (map (rewrite_simple sol en) args) = Some vs'
      ∧ map kind_of_value vs' = map kind_of_value vs.
Proof.
  intros Henv. induction args as [ | s args IH]; intros vs Hvs;
    simpl in Hvs.
  - injection Hvs as <-. exists []. split; reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-.
    destruct (IH ws eq_refl) as (ws' & Hws' & Hks).
    assert (Hv : ∃ v' : value,
              eval_simple vars' (rewrite_simple sol en s) = Some v'
              ∧ kind_of_value v' = kind_of_value v).
    { destruct s as [x | sym | c].
      - destruct (rewrite_simple_eval en vars vars' (Var x) v Henv I Hs)
          as (v' & H1 & H2 & _).
        exists v'. split; assumption.
      - injection Hs as <-. unfold rewrite_simple.
        destruct (is_symbol_used sol sym); eexists; split; reflexivity.
      - destruct (rewrite_simple_eval en vars vars' (Const c) v Henv I Hs)
          as (v' & H1 & H2 & _).
        exists v'. split; assumption. }
    destruct Hv as (v' & Hv' & Hk).
    exists (v' :: ws'). simpl. rewrite Hv', Hws'.
    split; [ reflexivity | simpl; rewrite Hk, Hks; reflexivity ].
Qed.

Lemma map_fst_rewrite (en : rebuild_env)
    (l : list (simple * flambda_kind)) :
  map fst (map (fun '(s, kind) => (rewrite_simple sol en s, kind)) l)
  = map (rewrite_simple sol en) (map fst l).
Proof.
  induction l as [ | [s k] l IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(* A binding of the first run, and maybe of the second. *)
Lemma env_ok_bind (kinds : finite_map flambda_kind) (vars vars' vars'' : env)
    (x : variable) (v : value) :
  env_ok kinds vars vars' ->
  (is_var_used sol x (kind_of_value v) = true ->
   ∃ v' : value, vars'' = (x, v') :: vars' ∧ rel v (Node_variable x) v') ->
  (vars'' = vars' ∨ ∃ w : value, vars'' = (x, w) :: vars') ->
  env_ok ((x, kind_of_value v) :: kinds) ((x, v) :: vars) vars''.
Proof.
  intros Henv Hx Hvars'' y w Hy. simpl in Hy |- *.
  destruct (Pos.eqb_spec y x) as [-> | Hne].
  - injection Hy as <-. split; [ reflexivity | ].
    intros Hused. destruct (Hx Hused) as (v' & -> & Hrel).
    exists v'. simpl. rewrite Pos.eqb_refl. auto.
  - destruct (Henv y w Hy) as [Hk Hused]. split; [ exact Hk | ].
    intros Hu. destruct (Hused Hu) as (v' & Hl & Hrel). exists v'.
    split; [ | exact Hrel ].
    destruct Hvars'' as [-> | (w' & ->)]; [ exact Hl | ].
    simpl. rewrite (proj2 (Pos.eqb_neq y x) Hne). exact Hl.
Qed.

(* ------------------------------------------------------------------------ *)
(* Static environments                                                      *)
(* ------------------------------------------------------------------------ *)

(* A block read off a value, in the two runs. *)
Lemma block_of_rel (statics statics' : static_env) (v v' : value) (n : node)
    (tag : Z) (fields : list value) :
  statics_rel statics statics' -> rel v n v' -> hu n ->
  block_of statics v = Some (tag, fields) ->
  ∃ (a : node) (fields' : list value),
    block_of statics' v' = Some (tag, fields')
    ∧ may_come_from sol a n
    ∧ fields_related (field_rel a) 0 fields fields'.
Proof.
  intros [_ [Hblocks _]] Hrel Hn Hb.
  destruct (block_of_cases _ _ _ _ Hb) as [-> | (s & -> & Hl)].
  - destruct Hrel as (a & fields' & Ha & -> & Hf).
    exists a, fields'. auto.
  - destruct Hrel as (-> & Ha & Hhu).
    destruct (Hblocks s tag fields Hl (Hhu Hn)) as (fields' & Hl' & Hf).
    exists (Node_symbol s), fields'.
    split; [ simpl; rewrite Hl'; reflexivity | auto ].
Qed.

Lemma block_of_none_rel (statics statics' : static_env) (v v' : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v n v' -> block_of statics v = None -> block_of statics' v' = None.
Proof.
  intros [Hnone [_ [Honly _]]] Hrel Hb.
  destruct v as [c | | | s | tag fields | elements | c | | ];
    cbn [rel] in Hrel; try contradiction.
  - destruct Hrel as [_ ->]. exact Hb.
  - destruct Hrel as (sites & vs' & _ & -> & _). reflexivity.
  - destruct Hrel as [-> _]. simpl in Hb |- *.
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl.
    + destruct (Honly s allocated Hl)
        as [(tag & fields & ->)
           | [(slot & syms & decls & vs & -> & [Hn | (vs' & Hs & _)] & _)
             | (Ho & _)]];
        [ discriminate Hb | rewrite Hn; reflexivity | rewrite Hs; exact Hb
        | ].
      destruct allocated as [ | | | c | bytes | els | ];
        try (exfalso; exact Ho); destruct Ho as [Hn | Ho];
        try (rewrite Hn; reflexivity);
        [ destruct Ho as (c' & Ho & _); rewrite Ho; reflexivity
        | rewrite Ho; reflexivity | ].
      destruct Ho as (els' & Ho & _). rewrite Ho. reflexivity.
    + rewrite (Hnone s Hl). reflexivity.
  - discriminate Hb.
  - destruct Hrel as (_ & elements' & -> & _). reflexivity.
  - destruct Hrel as (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - destruct Hrel as [_ (m & closure' & args' & -> & _)]. reflexivity.
Qed.

(* A symbol among the arguments of a primitive in the second run is
   bound if it is in the first. *)
Lemma bound_if_symbol_rel (statics statics' : static_env) (v v' : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v n v' -> hu n ->
  bound_if_symbol statics v = true -> bound_if_symbol statics' v' = true.
Proof.
  intros [_ [Hblocks [Honly _]]] Hrel Hn Hb.
  destruct v' as [ | | | s | | | | | ]; try reflexivity.
  pose proof (rel_symbol _ _ _ _ _ _ _ _ _ Hrel) as ->.
  destruct Hrel as (_ & _ & Hhu). simpl in Hb |- *.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
    [ | discriminate Hb ].
  destruct (Honly s allocated Hl)
    as [(tag & fields & ->)
       | [(slot & syms & decls & vs & -> & _ & Hs) | (_ & Hs)]].
  - destruct (Hblocks s tag fields Hl (Hhu Hn)) as (fields' & Hl' & _).
    rewrite Hl'. reflexivity.
  - destruct (lookup s (static_values statics')); [ reflexivity | ].
    exfalso. exact (Hs (Hhu Hn) eq_refl).
  - destruct (lookup s (static_values statics')); [ reflexivity | ].
    exfalso. exact (Hs (Hhu Hn) eq_refl).
Qed.

(* The fields of a block whose allocation has any usage are [sim]. *)
Lemma fields_sim (a : node) (k : nat) (fields fields' : list value) :
  au a -> fields_related (field_rel a) k fields fields' ->
  values_related sim fields fields'.
Proof.
  intros Hau. revert k fields'.
  induction fields as [ | field fields IH]; intros k [ | field' fields']
    Hfields; simpl in Hfields |- *; try contradiction; [ exact I | ].
  destruct Hfields as [(from & Hcons & Hused & _) Hrest]. split.
  - pose proof (block_field_not_local is_local Hlocal k Value) as Hl.
    apply (rel_au_sim G sol keep called (exact = true) True is_local Hc Hlocal
      field from
             field');
      [ exact (any_usage_constructor _ _ _ Hc _ _ _ Hau Hcons Hl) | ].
    apply Hused. exact (field_used_any_usage _ _ _ Hc _ _ _ Hcons Hau Hl).
  - exact (IH _ _ Hrest).
Qed.

(* What a boxed number holds in the second run: the same as in the
   first, or what is not a constant in either. *)
Lemma boxed_contents_cases (c c' : value) :
  c' = c ∨ sim c c' -> c' = c ∨ ∀ k : const, c <> Value_const k.
Proof.
  intros [-> | Hsim]; [ left; reflexivity | ].
  destruct c; cbn [sim] in Hsim;
    try (left; exact Hsim); right; intros k Hk; discriminate Hk.
Qed.

(* What a symbol with any usage is bound to in the two runs. *)
Lemma statics_au (statics statics' : static_env) (s : symbol) :
  statics_rel statics statics' ->
  au (Node_symbol s) ->
  (lookup s (static_values statics) = None ->
   lookup s (static_values statics') = None)
  ∧ (∀ (tag : Z) (fields : list value),
       lookup s (static_values statics) = Some (Static_block tag fields) ->
       ∃ fields' : list value,
         lookup s (static_values statics') = Some (Static_block tag fields')
         ∧ values_related sim fields fields')
  ∧ (∀ (slot : function_slot) (syms : list (function_slot * symbol))
       (decls : list (function_slot * code_id))
       (vs : list (value_slot * value)),
       lookup s (static_values statics)
       = Some (Static_closure_of_set slot syms decls vs) ->
       ∃ vs' : list (value_slot * value),
         lookup s (static_values statics')
         = Some (Static_closure_of_set slot syms decls vs'))
  ∧ (∀ allocated : statically_allocated_value,
       lookup s (static_values statics) = Some allocated ->
       (∃ (tag : Z) (fields : list value),
          allocated = Static_block tag fields)
       ∨ (∃ (slot : function_slot) (syms : list (function_slot * symbol))
            (decls : list (function_slot * code_id))
            (vs : list (value_slot * value)),
            allocated = Static_closure_of_set slot syms decls vs)
       ∨ (∃ c c' : value,
            allocated = Static_boxed_number c
            ∧ lookup s (static_values statics')
              = Some (Static_boxed_number c')
            ∧ (c' = c ∨ sim c c'))
       ∨ (∃ bytes : list Z,
            allocated = Static_string bytes
            ∧ lookup s (static_values statics') = Some allocated)
       ∨ ∃ elements elements' : list value,
           allocated = Static_immutable_array elements
           ∧ lookup s (static_values statics')
             = Some (Static_immutable_array elements')
           ∧ Forall2 sim elements elements').
Proof.
  intros [Hnone [Hblocks [Honly _]]] Hau.
  pose proof (au_hu _ _ _ Hc _ Hau) as Hhu.
  split; [ exact (Hnone s) | split; [ | split ] ].
  - intros tag fields Hl.
    destruct (Hblocks s tag fields Hl Hhu) as (fields' & Hl' & Hf).
    exists fields'. split; [ exact Hl' | exact (fields_sim _ _ _ _ Hau Hf) ].
  - intros slot syms decls vs Hl.
    destruct (Honly s _ Hl)
      as [(tag & fields & Heq)
         | [(slot' & syms' & decls' & vs0 & Heq & Hs & Hb) | ([] & _)]];
      [ discriminate Heq | ].
    injection Heq as <- <- <- <-.
    destruct Hs as [Hn | (vs' & Hs & _)]; [ | exists vs'; exact Hs ].
    exfalso. exact (Hb Hhu Hn).
  - intros allocated Hl.
    destruct (Honly s _ Hl)
      as [Hb | [(slot & syms & decls & vs & Heq & _) | (Ho & Hbound)]];
      [ left; exact Hb
      | right; left; exists slot, syms, decls, vs; exact Heq | ].
    right. right.
    destruct allocated as [ | | | c | bytes | els | ];
      try (exfalso; exact Ho);
      (destruct Ho as [Hn | Ho]; [ destruct (Hbound Hhu Hn) | ]).
    + left. destruct Ho as (c' & Hs & Hc'). exists c, c'.
      split; [ reflexivity | split; [ exact Hs | exact Hc' ] ].
    + right. left. exists bytes. split; [ reflexivity | exact Ho ].
    + right. right. destruct Ho as (els' & Hs & Hf). exists els, els'.
      split; [ reflexivity | split; [ exact Hs | ] ].
      exact (Forall2_impl _ element_rel_sim (Hf Hau)).
Qed.

(* ------------------------------------------------------------------------ *)
(* The fields of a block                                                    *)
(* ------------------------------------------------------------------------ *)

(* The fields of a block made at [base], from field [k]: a field used is
   related; one not used is poison. *)
Lemma rebuild_fields_eval (en : rebuild_env) (vars vars' : env)
    (base : node) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (fields : list simple) (k : nat) (vs : list value),
    (∀ (i : nat) (s : simple),
       nth_error fields i = Some s ->
       g_constructor G base (Field_block (k + i) Value) (node_of_simple s)) ->
    simples_noted noted fields ->
    eval_simples vars fields = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (rebuild_fields sol en base k fields) = Some vs'
      ∧ fields_related (field_rel base) k vs vs'
      ∧ ∀ (i : nat) (v v' : value),
          nth_error vs i = Some v -> nth_error vs' i = Some v' ->
          v' = Value_const (Poison Value)
          ∨ ∃ n : node, rel v n v' ∧ hu n.
Proof.
  intros Henv. induction fields as [ | s fields IH]; intros k vs Hcons Hnoted
    Hvs; simpl in Hvs.
  - injection Hvs as <-. exists [].
    split; [ reflexivity | split; [ exact I | ] ].
    intros [ | i]; discriminate.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars fields) as [ws | ] eqn:Hws;
      [ | discriminate ].
    injection Hvs as <-.
    assert (Hs_cons : g_constructor G base (Field_block k Value)
                        (node_of_simple s)).
    { rewrite <- (Nat.add_0_r k). exact (Hcons 0 s eq_refl). }
    assert (Hcons' : ∀ (i : nat) (s' : simple),
              nth_error fields i = Some s' ->
              g_constructor G base (Field_block (S k + i) Value)
                (node_of_simple s')).
    { intros i s' Hi. replace (S k + i) with (k + S i) by lia.
      exact (Hcons (S i) s' Hi). }
    destruct (IH (S k) ws Hcons' (fun s' Hin => Hnoted s' (or_intror Hin))
                eq_refl)
      as (ws' & Hws' & Hf & Hpoint).
    destruct (s_field_used sol base (Field_block k Value)) eqn:Hused.
    + pose proof (field_used_hu G sol is_local Hc Hexact base
                    (node_of_simple s) _
                    (block_field_not_local is_local Hlocal k Value)
                    Hused Hs_cons) as Hhu.
      destruct (rewrite_simple_eval en vars vars' s v Henv
                  (Hnoted s (or_introl eq_refl)) Hs)
        as (v' & Hs' & _ & Hkept).
      destruct (Hkept (kept_of_hu en s Hhu)) as [_ Hrel].
      exists (v' :: ws'). cbn [rebuild_fields eval_simples].
      unfold field_used. rewrite Hused, Hs', Hws'.
      split; [ reflexivity | split ].
      * split; [ | exact Hf ].
        exists (node_of_simple s).
        split; [ exact Hs_cons | split; [ auto | intros E; congruence ] ].
      * intros [ | i] w w' Hw Hw'; simpl in Hw, Hw';
          [ | exact (Hpoint i w w' Hw Hw') ].
        injection Hw as <-. injection Hw' as <-. right.
        exists (node_of_simple s). auto.
    + exists (Value_const (Poison Value) :: ws').
      cbn [rebuild_fields eval_simples]. unfold field_used.
      rewrite Hused, Hws'.
      split; [ reflexivity | split ].
      * split; [ | exact Hf ].
        exists (node_of_simple s).
        split;
          [ exact Hs_cons | split; [ intros Hu; congruence | reflexivity ] ].
      * intros [ | i] w w' Hw Hw'; simpl in Hw, Hw';
          [ | exact (Hpoint i w w' Hw Hw') ].
        injection Hw' as <-. left. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Primitives                                                               *)
(* ------------------------------------------------------------------------ *)

(* A unary primitive on a simple that the rebuild keeps. *)
Lemma unary_target (en : rebuild_env) (vars' : env) (statics' : static_env)
    (up : unary_primitive) (s1 : simple) (v1' : value) :
  rewrite_simple sol en s1 = s1 -> eval_simple vars' s1 = Some v1' ->
  eval_named statics' vars' (Prim (Unary up) (map (rewrite_simple sol en) [s1]))
  = eval_unary_primitive statics' up v1'.
Proof.
  intros Hr Hs. cbn [map eval_named eval_simples]. rewrite Hr, Hs.
  cbv beta iota. apply eval_prim_unary.
Qed.

(* A load of a field of a block, into a variable with a use. *)
Lemma block_load_rel (statics statics' : static_env) (x : variable)
    (base : node) (kind : block_access_kind) (fld : Z) (v1 v1' v : value) :
  statics_rel statics statics' ->
  g_accessor G (Node_variable x) (Field_block (Z.to_nat fld) Value) base ->
  rel v1 base v1' -> hu (Node_variable x) ->
  eval_block_load statics kind fld v1 = Some v ->
  ∃ v' : value,
    eval_block_load statics' kind fld v1' = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Hst Hacc Hrel Hx Hev.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc) as Hbase.
  destruct kind as [tag size fk]. unfold eval_block_load in Hev |- *.
  destruct (block_of statics v1) as [[btag fields] | ] eqn:Hb;
    [ | discriminate Hev ].
  destruct (block_of_rel statics statics' v1 v1' base btag fields Hst Hrel
              Hbase Hb)
    as (a & fields' & Hb' & Ha & Hf).
  rewrite Hb'. cbv beta iota in Hev |- *.
  rewrite (fields_related_length _ _ _ _ Hf).
  destruct (known_is tag btag && known_is size (Z.of_nat (length fields))
            && (0 <=? fld)%Z);
    [ | discriminate Hev ].
  destruct (nth_error fields (Z.to_nat fld)) as [loaded | ] eqn:Hnth;
    [ | discriminate Hev ].
  destruct (fields_related_nth _ _ _ _ _ _ Hf Hnth)
    as (loaded' & Hnth' & from & Hcons & Hused & _).
  rewrite Hnth'.
  pose proof (block_field_not_local is_local Hlocal (Z.to_nat fld) Value)
    as Hl.
  pose proof (field_used_read is_local G sol Hc a base (Node_variable x) from
                _ Hl Hcons Hacc Ha Hx) as Hu.
  pose proof (rel_read G sol keep called (exact = true) True is_local Hc _ _ _
      _ _
    _ _ Hl
                Hcons Hacc Ha (Hused Hu)) as Hrel'.
  exists loaded'.
  assert (Hcond :
    match fk with
    | Any_value => kind_eqb (kind_of_value loaded') Value
    | Immediate => is_tagged_immediate loaded'
    end
    = match fk with
      | Any_value => kind_eqb (kind_of_value loaded) Value
      | Immediate => is_tagged_immediate loaded
      end).
  { destruct fk;
      [ rewrite (rel_kind _ _ _ _ _ _ _ _ _ Hrel')
      | rewrite (rel_tagged _ _ _ _ _ _ _ _ _ Hrel') ];
      reflexivity. }
  rewrite Hcond.
  destruct (match fk with
            | Any_value => kind_eqb (kind_of_value loaded) Value
            | Immediate => is_tagged_immediate loaded
            end);
    [ | discriminate Hev ].
  injection Hev as <-. split; [ reflexivity | exact Hrel' ].
Qed.

(* The fields of a tuple, as a call of tupled code reads them, of a value
   related at a node that has a use: the second run has as many, of the
   same kinds, all values, and one read into a node that has a use is
   related there. *)
Lemma tuple_fields_rel (statics statics' : static_env) (size : nat)
    (tuple tuple' : value) (n : node) (fields : list value) :
  statics_rel statics statics' -> rel tuple n tuple' -> hu n ->
  tuple_fields statics size tuple = Some fields ->
  ∃ fields' : list value,
    tuple_fields statics' size tuple' = Some fields'
    ∧ Forall (fun v => kind_of_value v = Value) fields
    ∧ map kind_of_value fields' = map kind_of_value fields
    ∧ ∀ (i : nat) (p : node) (f f' : value),
        g_accessor G p (Field_block i Value) n -> hu p ->
        nth_error fields i = Some f -> nth_error fields' i = Some f' ->
        rel f p f'.
Proof.
  intros Hst Hrel Hn Hf.
  destruct size as [ | size].
  { injection Hf as <-. exists [].
    split; [ reflexivity | split; [ constructor | split; [ reflexivity | ] ] ].
    intros [ | i] p f f' _ _ Hi; discriminate Hi. }
  destruct (proj1 (FlambdaTupledProofs.tuple_is_block statics (S size) tuple
                     fields ltac:(discriminate)) Hf)
    as (Hb & Hlen & Hkinds).
  destruct (block_of_rel statics statics' tuple tuple' n 0%Z fields Hst Hrel
              Hn Hb)
    as (a & fields' & Hb' & Ha & Hfr).
  assert (Hkinds' : Forall (fun v => kind_of_value v = Value) fields').
  { apply Forall_forall. intros v' Hin. apply In_nth_error in Hin as [i Hi].
    destruct (fields_related_nth' _ _ _ _ _ _ Hfr Hi)
      as (v & Hv & from & _ & Hused & Hpoison).
    destruct (s_field_used sol a (Field_block i Value)) eqn:Hu.
    - rewrite (rel_kind _ _ _ _ _ _ _ _ _ (Hused Hu)).
      rewrite Forall_forall in Hkinds. exact (Hkinds v (nth_error_In _ _ Hv)).
    - rewrite (Hpoison Hu). reflexivity. }
  assert (Hlen' : length fields' = length fields)
    by exact (fields_related_length _ _ _ _ Hfr).
  exists fields'. split.
  { apply (proj2 (FlambdaTupledProofs.tuple_is_block statics' (S size) tuple'
                    fields' ltac:(discriminate))).
    split; [ exact Hb' | split; [ rewrite Hlen'; exact Hlen | exact Hkinds' ] ].
  }
  split; [ exact Hkinds | split ].
  { clear - Hkinds Hkinds' Hlen'. revert fields' Hkinds' Hlen'.
    induction Hkinds as [ | v l Hv _ IH]; intros [ | v' l'] Hk' Hl;
      simpl in Hl |- *; try discriminate Hl; [ reflexivity | ].
    apply Forall_cons_iff in Hk' as [Hv' Hk''].
    rewrite Hv, Hv', (IH l' Hk'' ltac:(lia)). reflexivity. }
  intros i p f f' Hacc Hp Hfi Hfi'.
  destruct (fields_related_nth _ _ _ _ _ _ Hfr Hfi)
    as (f'' & Hf'' & from & Hcons & Hused & _).
  rewrite Hfi' in Hf''. injection Hf'' as <-.
  pose proof (block_field_not_local is_local Hlocal i Value) as Hl.
  pose proof (field_used_read is_local G sol Hc a n p from _ Hl Hcons Hacc Ha
                Hp) as Hu.
  exact (rel_read G sol keep called (exact = true) True is_local Hc _ _ _ _ _
           _ _ Hl Hcons Hacc Ha (Hused Hu)).
Qed.

(* [Is_int] of a variant, whose argument has a use. *)
Lemma is_int_rel (statics statics' : static_env) (v1 v1' : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v1 n v1' -> hu n ->
  eval_is_int statics' true v1' = eval_is_int statics true v1.
Proof.
  intros Hst Hrel Hn. unfold eval_is_int.
  rewrite (rel_tagged _ _ _ _ _ _ _ _ _ Hrel).
  destruct (is_tagged_immediate v1); [ reflexivity | ].
  destruct (block_of statics v1) as [[tag fields] | ] eqn:Hb.
  - destruct (block_of_rel statics statics' v1 v1' n tag fields Hst Hrel Hn Hb)
      as (a & fields' & Hb' & _).
    rewrite Hb'. reflexivity.
  - rewrite (block_of_none_rel statics statics' v1 v1' n Hst Hrel Hb).
    reflexivity.
Qed.

(* [Get_tag], whose argument has a use. *)
Lemma get_tag_rel (statics statics' : static_env) (v1 v1' v : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v1 n v1' -> hu n ->
  eval_unary_primitive statics Get_tag v1 = Some v ->
  eval_unary_primitive statics' Get_tag v1' = Some v.
Proof.
  intros Hst Hrel Hn Hev. cbn [eval_unary_primitive] in Hev |- *.
  destruct (block_of statics v1) as [[tag fields] | ] eqn:Hb;
    [ | discriminate Hev ].
  destruct (block_of_rel statics statics' v1 v1' n tag fields Hst Hrel Hn Hb)
    as (a & fields' & Hb' & _).
  rewrite Hb'. exact Hev.
Qed.

(* The unboxing of a number, into a variable with a use: the number is
   used, and kept. *)
Lemma unbox_rel (statics statics' : static_env) (x : variable) (base : node)
    (bn : boxable_number) (v1 v1' v : value) :
  statics_rel statics statics' ->
  g_accessor G (Node_variable x) (Field_boxed_number bn) base ->
  rel v1 base v1' -> hu (Node_variable x) ->
  eval_unary_primitive statics (Unbox_number bn) v1 = Some v ->
  eval_unary_primitive statics' (Unbox_number bn) v1' = Some v.
Proof.
  intros Hst Hacc Hrel Hx Hev. cbn [eval_unary_primitive] in Hev |- *.
  destruct (boxed_number_of statics v1) as [c | ] eqn:Hbx;
    [ | discriminate Hev ].
  destruct (boxed_number_of_cases _ _ _ Hbx) as [-> | (s & -> & Hl)].
  - destruct Hrel as (a & bn' & from & Ha & Hk & Hcons & ->).
    destruct (is_number_of_kind bn c) eqn:Hkind; [ | discriminate Hev ].
    pose proof (is_number_of_kind_unique _ _ _ Hk Hkind) as Heq.
    subst bn'.
    rewrite (field_used_read is_local G sol Hc a base (Node_variable x) from
               _ (boxed_field_not_local is_local Hlocal bn) Hcons Hacc Ha Hx).
    change (boxed_number_of statics' (Value_boxed_number c))
      with (boxed_number_of statics (Value_boxed_number c)).
    rewrite Hbx. cbv beta iota. rewrite Hkind. exact Hev.
  - destruct (proj1 (proj2 (proj2 Hst)) s _ Hl)
      as [(tag & fields & Heq) | [(slot & syms & decls & vs & Heq & _)
                                 | ([Hn | Hs'] & Hbound)]];
      [ discriminate Heq | discriminate Heq | | ].
    + destruct Hrel as (_ & _ & Hhu).
      destruct (Hbound (Hhu (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc)) Hn).
    + destruct Hrel as (-> & _ & _).
      destruct Hs' as (c' & Hs' & Hc').
      assert (Hcc : c' = Value_const c) by (destruct Hc' as [H | H]; exact H).
      subst c'.
      assert (Hbx' : boxed_number_of statics' (Value_symbol s)
                     = boxed_number_of statics (Value_symbol s))
        by (simpl; rewrite Hs', Hl; reflexivity).
      rewrite Hbx', Hbx. exact Hev.
Qed.

(* The boxing of a number: kept with its contents if the field of the
   number is used, poison otherwise. *)
Lemma box_number_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (bn : boxable_number)
    (s1 : simple) (v1 v : value) :
  env_ok (env_kinds en) vars vars' -> env_flows_ vars ->
  g_constructor G (Node_variable x) (Field_boxed_number bn)
    (node_of_simple s1) ->
  simple_noted noted s1 ->
  eval_simple vars s1 = Some v1 ->
  eval_unary_primitive statics (Box_number bn) v1 = Some v ->
  ∃ v' : value,
    eval_named statics' vars'
      (rebuild_named sol en x (Prim (Unary (Box_number bn)) [s1]))
    = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hvars Hcons Hnoted Hs1 Hev. cbn [eval_unary_primitive] in Hev.
  destruct v1 as [c | | | | | | | | ]; try discriminate Hev.
  destruct (is_number_of_kind bn c) eqn:Hk; [ | discriminate Hev ].
  injection Hev as <-.
  pose proof (simple_flows is_local G sol noted cds keep no_loc Hc Hconsts
      Hsyms _ _ _
                Hvars Hnoted Hs1) as Hflows1.
  assert (Hself : may_come_from sol (Node_variable x) (Node_variable x)).
  { exact (alloc_self is_local G sol Hc _ _ _ Hcons
             (as_hs _ _ _ Hc _ Hflows1)). }
  cbn [rebuild_named]. unfold field_used.
  destruct (s_field_used sol (Node_variable x) (Field_boxed_number bn))
    eqn:Hused.
  - pose proof (field_used_hu G sol is_local Hc Hexact _ _ _
                  (boxed_field_not_local is_local Hlocal bn) Hused Hcons)
      as Hhu.
    destruct (rewrite_simple_eval en vars vars' s1 _ Henv Hnoted Hs1)
      as (v1' & Hs1' & _ & Hkept).
    destruct (Hkept (kept_of_hu en s1 Hhu)) as [Hr Hrel].
    destruct Hrel as [_ ->]. rewrite Hr in Hs1'.
    exists (Value_boxed_number c). split.
    + rewrite (unary_target en vars' statics' (Box_number bn) s1 _ Hr Hs1').
      cbn [eval_unary_primitive]. rewrite Hk. reflexivity.
    + exists (Node_variable x), bn, (node_of_simple s1). rewrite Hused. auto.
  - exists (Value_const (Poison Value)). split; [ reflexivity | ].
    exists (Node_variable x), bn, (node_of_simple s1). rewrite Hused. auto.
Qed.

(* A block made: its fields used are related. *)
Lemma make_block_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (tag : Z)
    (fields : list simple) (vs : list value) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' ->
  block_edges G noted (Node_variable x) fields ->
  eval_simples vars fields = Some vs ->
  eval_variadic_primitive statics (Make_block (Block_kind_values tag)) vs
  = Some v ->
  ∃ v' : value,
    eval_named statics' vars'
      (Prim (Variadic (Make_block (Block_kind_values tag)))
         (rebuild_fields sol en (Node_variable x) 0 fields))
    = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst (Hcons & His & Hgt & Hnoted) Hvs Hev.
  cbn [eval_variadic_primitive] in Hev.
  destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs)
    eqn:Hkinds; [ | discriminate Hev ].
  destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
    [ | discriminate Hev ].
  injection Hev as <-.
  destruct (rebuild_fields_eval en vars vars' (Node_variable x) Henv fields 0
              vs Hcons Hnoted Hvs)
    as (vs' & Hvs' & Hf & Hpoint).
  rewrite forallb_forall in Hkinds, Hbound.
  exists (Value_block tag vs'). split.
  - cbn [eval_named]. rewrite Hvs'. cbv beta iota.
    rewrite eval_prim_variadic. cbn [eval_variadic_primitive].
    assert (Hkinds' : forallb (fun v => kind_eqb (kind_of_value v) Value) vs'
                      = true).
    { apply forallb_forall. intros v' Hin. apply In_nth_error in Hin as [i Hi].
      destruct (fields_related_nth' _ _ _ _ _ _ Hf Hi) as (w & Hw & _).
      destruct (Hpoint i w v' Hw Hi) as [-> | (n & Hrel & _)];
        [ reflexivity | ].
      rewrite (rel_kind _ _ _ _ _ _ _ _ _ Hrel).
      exact (Hkinds w (nth_error_In _ _ Hw)). }
    assert (Hbound' : forallb (bound_if_symbol statics') vs' = true).
    { apply forallb_forall. intros v' Hin. apply In_nth_error in Hin as [i Hi].
      destruct (fields_related_nth' _ _ _ _ _ _ Hf Hi) as (w & Hw & _).
      destruct (Hpoint i w v' Hw Hi) as [-> | (n & Hrel & Hn)];
        [ reflexivity | ].
      exact (bound_if_symbol_rel statics statics' w v' n Hst Hrel Hn
               (Hbound w (nth_error_In _ _ Hw))). }
    rewrite Hkinds', Hbound'. reflexivity.
  - exists (Node_variable x), vs'.
    split; [ | split; [ reflexivity | exact Hf ] ].
    exact (alloc_self is_local G sol Hc _ _ _ His
             (as_hs _ _ _ Hc _ (consts_any_source is_local G sol Hc Hconsts))).
Qed.
Lemma rewrite_simple_au (en : rebuild_env) (vars vars' : env) (s : simple)
    (w : value) :
  env_ok (env_kinds en) vars vars' ->
  (∃ c : const, s = Const c) ∨ au (node_of_simple s) ->
  eval_simple vars s = Some w ->
  ∃ w' : value,
    eval_simple vars' (rewrite_simple sol en s) = Some w'
    ∧ element_rel w w'.
Proof.
  intros Henv Hs Hw. destruct s as [y | sym | c]; simpl in Hw, Hs.
  - destruct Hs as [(c & Heq) | Hau]; [ discriminate Heq | ].
    destruct (Henv y w Hw) as [_ Hused].
    unfold rewrite_simple. cbv zeta.
    rewrite (env_ok_kind en vars vars' y w Henv Hw).
    rewrite (is_var_used_hu y _ (au_hu _ _ _ Hc _ Hau)).
    destruct (Hused (is_var_used_hu y _ (au_hu _ _ _ Hc _ Hau)))
      as (w' & Hl & Hrel).
    exists w'. split; [ exact Hl | right; exists (Node_variable y); auto ].
  - destruct Hs as [(c & Heq) | Hau]; [ discriminate Heq | ].
    unfold rewrite_simple, is_symbol_used. rewrite (au_hu _ _ _ Hc _ Hau).
    exists w. split; [ exact Hw | ]. injection Hw as <-.
    left. split; [ reflexivity | right; exists sym; auto ].
  - exists w. split; [ exact Hw | ]. injection Hw as <-.
    left. split; [ reflexivity | left; exists c; reflexivity ].
Qed.

Lemma rewrite_simples_au (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (args : list simple) (vs : list value),
    (∀ s : simple, s ∈ args ->
       (∃ c : const, s = Const c) ∨ au (node_of_simple s)) ->
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (map (rewrite_simple sol en) args) = Some vs'
      ∧ Forall2 element_rel vs vs'.
Proof.
  intros Henv. induction args as [ | s args IH]; intros vs Hargs Hvs;
    simpl in Hvs |- *.
  - injection Hvs as <-. exists []. split; [ reflexivity | constructor ].
  - destruct (eval_simple vars s) as [w | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-.
    destruct (rewrite_simple_au en vars vars' s w Henv
                (Hargs s (or_introl eq_refl)) Hs) as (w' & Hs' & Hw).
    destruct (IH ws (fun s' Hin => Hargs s' (or_intror Hin)) eq_refl)
      as (ws' & Hws' & Hall).
    exists (w' :: ws'). rewrite Hs', Hws'.
    split; [ reflexivity | constructor; assumption ].
Qed.

Lemma Forall2_values_related {R : value -> value -> Prop} (l l' : list value) :
  Forall2 R l l' -> values_related R l l'.
Proof. induction 1; simpl; auto. Qed.

Lemma values_related_Forall2 {R : value -> value -> Prop} (l l' : list value) :
  values_related R l l' -> Forall2 R l l'.
Proof.
  revert l'. induction l as [ | w l IH]; intros [ | w' l'] H; simpl in H;
    try contradiction; constructor;
    [ exact (proj1 H) | exact (IH _ (proj2 H)) ].
Qed.

Ltac obs_close :=
  simpl; intros;
  first [ discriminate | reflexivity | congruence
        | (match goal with
           | H : context [lookup ?a ?b] |- _ => destruct (lookup a b)
           end; congruence) ].

(* [sim] relates what the primitives observe of a value, as
   FlambdaPrimitiveSpec.eval_prim_related asks. *)
Lemma sim_respects (statics statics' : static_env) :
  statics_rel statics statics' -> respects_observations sim statics statics'.
Proof.
  intros Hst. split; [ split | split; [ split; [ | split ] | ] ].
  - intros c v' H. exact H.
  - intros v c H.
    destruct v as [c' | code slots | slot decls vs | s | tag fields | elements
                  | c' | l | callee args arity];
      cbn [sim] in H; try contradiction.
    + exact (eq_sym H).
    + destruct H as (vs' & Heq & _). discriminate Heq.
    + destruct H as [Heq _]. discriminate Heq.
    + destruct H as (fields' & Heq & _). discriminate Heq.
    + destruct H as (elements' & Heq & _). discriminate Heq.
    + discriminate H.
    + destruct H as [_ (closure' & args' & Heq & _)]. discriminate Heq.
  - intros c. split; reflexivity.
  - intros tag fields fields' H. exists fields'.
    split; [ reflexivity | exact (Forall2_values_related _ _ H) ].
  - intros elements elements' H. exists elements'.
    split; [ reflexivity | exact (Forall2_values_related _ _ H) ].
  - intros v v' Hsim.
    destruct v as [c | code slots | slot decls vs | s | tag fields | elements
                  | c | l | callee args arity];
      cbn [sim] in Hsim; try contradiction.
    + subst v'. repeat split; obs_close.
    + destruct Hsim as (vs' & -> & _). repeat split; obs_close.
    + destruct Hsim as [-> Hau].
      destruct (statics_au statics statics' s Hst Hau)
        as (Hnone & Hblk & Hclo & Honly).
      destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl.
      * destruct (Honly allocated eq_refl)
          as [(t0 & f0 & ->)
             | [(sl & sy & de & vs0 & ->)
               | [(c & c' & -> & Hl' & Hc')
                 | [(bytes & -> & Hl') | (els & els' & -> & Hl' & Hf)]]]].
        -- destruct (Hblk t0 f0 eq_refl) as (fields' & Hl' & Hf).
           repeat split; simpl; rewrite ?Hl, ?Hl'; try obs_close.
           intros t f Hb. injection Hb as <- <-. exists fields'.
           split; [ reflexivity | exact (values_related_Forall2 _ _ Hf) ].
        -- destruct (Hclo sl sy de vs0 eq_refl) as (vs' & Hl').
           repeat split; simpl; rewrite ?Hl, ?Hl'; obs_close.
        -- destruct (boxed_contents_cases c c' Hc') as [-> | Hnc];
             [ repeat split; simpl; rewrite ?Hl, ?Hl'; obs_close | ].
           repeat split; simpl; rewrite ?Hl, ?Hl'; try obs_close.
           intros c0 Hc0. exfalso.
           destruct c; simpl in Hc0; try discriminate Hc0.
           exact (Hnc _ eq_refl).
        -- repeat split; simpl; rewrite ?Hl, ?Hl'; obs_close.
        -- repeat split; simpl; rewrite ?Hl, ?Hl'; try obs_close.
           intros e Ha. injection Ha as <-. exists els'.
           split; [ reflexivity | exact Hf ].
      * pose proof (Hnone eq_refl) as Hl'.
        repeat split; simpl; rewrite ?Hl, ?Hl'; obs_close.
    + destruct Hsim as (fields' & -> & Hf).
      repeat split; try obs_close.
      intros t f Hb. simpl in Hb. injection Hb as <- <-. exists fields'.
      split; [ reflexivity | exact (values_related_Forall2 _ _ Hf) ].
    + destruct Hsim as (elements' & -> & Hf).
      repeat split; try obs_close.
      intros e Ha. simpl in Ha. injection Ha as <-. exists elements'.
      split; [ reflexivity | exact (values_related_Forall2 _ _ Hf) ].
    + subst v'. repeat split; obs_close.
    + destruct Hsim as [_ (closure' & args' & -> & _)].
      repeat split; obs_close.
Qed.

(* The primitives of the default case: a constant, the identity, a load
   from an immutable array and the making of one. *)
Lemma default_prim_cases (statics : static_env) (p : prim)
    (args : list simple) (vs : list value) (v : value) :
  prim_classified p args = false ->
  length vs = length args ->
  eval_prim statics p vs = Some v ->
  (∃ c : const, v = Value_const c)
  ∨ (∃ k : flambda_kind, p = Unary (Opaque_identity k) ∧ vs = [v])
  ∨ (∃ (ak : array_kind) (lk : array_load_kind) (mut : mutability)
       (arr idx : value) (elements : list value),
       p = Binary (Array_load ak lk mut) ∧ vs = [arr; idx]
       ∧ array_of statics arr = Some elements
       ∧ array_loaded lk elements idx = Some v)
  ∨ (∃ ak : array_kind,
       p = Variadic (Make_immutable_array ak)
       ∧ v = Value_immutable_array vs).
Proof.
  intros Hcl Hlen Heval.
  destruct (eval_prim_arity _ _ _ _ Heval)
    as [(p' & v1 & -> & -> & Hu) | [(p' & v1 & v2 & -> & -> & Hb)
                                  | (p' & -> & Hv)]].
  - destruct args as [ | s1 [ | s2 args]]; simpl in Hlen; try discriminate Hlen.
    destruct p' as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                   | | bn | bn | | | r];
      simpl in Hcl; try discriminate Hcl; cbn [eval_unary_primitive] in Hu;
      try (const_value Hu).
    destruct (kind_eqb (kind_of_value v1) ok && bound_if_symbol statics v1);
      [ | discriminate Hu ].
    injection Hu as ->. right. left. exists ok. auto.
  - destruct p' as [ak lk mut' | width | op | ik op | ik op | ik beh];
      cbn [eval_binary_primitive] in Hb; try (const_value Hb).
    destruct (array_of statics v1) as [elements | ] eqn:Harr;
      [ | discriminate Hb ].
    right. right. left. exists ak, lk, mut', v1, v2, elements. auto.
  - destruct p' as [bk | ak]; [ discriminate Hcl | ].
    cbn [eval_variadic_primitive] in Hv.
    destruct (array_made ak vs) as [elements | ] eqn:Hmade;
      [ | discriminate Hv ].
    destruct (array_made_of_arguments _ _ _ Hmade) as [-> _].
    destruct (forallb (bound_if_symbol statics) vs); [ | discriminate Hv ].
    injection Hv as <-. right. right. right. exists ak. auto.
Qed.

Lemma sim_const_r (v : value) (c : const) :
  sim v (Value_const c) -> v = Value_const c.
Proof.
  destruct v; cbn [sim]; try contradiction; try (intros H; discriminate H);
    try (intros [H _]; discriminate H);
    try (intros (? & H & _); discriminate H);
    try (intros [_ (? & ? & H & _)]; discriminate H).
  intros H. exact (eq_sym H).
Qed.

Lemma values_related_length {R : value -> value -> Prop}
    (l l' : list value) :
  values_related R l l' -> length l' = length l.
Proof.
  revert l'. induction l as [ | w l IH]; intros [ | w' l'] H; simpl in H;
    try contradiction; [ reflexivity | ].
  simpl. f_equal. exact (IH _ (proj2 H)).
Qed.

Lemma values_related_nth {R : value -> value -> Prop} (l l' : list value)
    (i : nat) (w w' : value) :
  values_related R l l' -> nth_error l i = Some w ->
  nth_error l' i = Some w' -> R w w'.
Proof.
  revert l' i. induction l as [ | a0 l IH]; intros [ | a0' l'] i H Hn Hn';
    simpl in H; try contradiction; [ destruct i; discriminate | ].
  destruct i as [ | i]; simpl in Hn, Hn'.
  - injection Hn as <-. injection Hn' as <-. exact (proj1 H).
  - exact (IH l' i (proj2 H) Hn Hn').
Qed.

(* The loads at one index from two arrays of related elements. *)
Lemma array_loaded_related (R : value -> value -> Prop)
    (lk : array_load_kind) (els els' : list value) (idx v v' : value) :
  values_related R els els' ->
  array_loaded lk els idx = Some v -> array_loaded lk els' idx = Some v' ->
  R v v'.
Proof.
  intros Hr Hl Hl'. unfold array_loaded, array_index in Hl, Hl'.
  rewrite (values_related_length _ _ Hr) in Hl'.
  destruct (int_of_value Standard_tagged_immediate idx) as [i | ];
    [ | discriminate Hl ].
  cbv beta iota in Hl, Hl'.
  destruct ((0 <=? i)%Z && (i <? Z.of_nat (length els))%Z);
    [ | discriminate Hl ].
  cbv beta iota in Hl, Hl'.
  destruct (nth_error els (Z.to_nat i)) as [w | ] eqn:Hn;
    [ | discriminate Hl ].
  destruct (nth_error els' (Z.to_nat i)) as [w' | ] eqn:Hn';
    [ | discriminate Hl' ].
  cbv beta iota in Hl, Hl'.
  destruct (array_element_ok _ w); [ injection Hl as <- | discriminate Hl ].
  destruct (array_element_ok _ w'); [ injection Hl' as <- | discriminate Hl' ].
  exact (values_related_nth _ _ _ _ _ Hr Hn Hn').
Qed.

(* A primitive of the default case, whose variable has a use: its
   arguments have any usage, so the second run gives it [sim] values, and
   it has a related value. *)
Lemma default_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (p : prim)
    (args : list simple) (vs : list value) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' ->
  default_prim_edges G x args -> prim_classified p args = false ->
  hu (Node_variable x) ->
  eval_simples vars args = Some vs -> eval_prim statics p vs = Some v ->
  ∃ v' : value,
    eval_named statics' vars' (Prim p (map (rewrite_simple sol en) args))
    = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst [Has Huse] Hcl Hx Hvs Hev.
  pose proof (any_source_input _ _ _ Hc _ Has) as Hasx.
  assert (Hargs : ∀ s : simple, s ∈ args ->
            (∃ c : const, s = Const c) ∨ au (node_of_simple s)).
  { intros s Hin. destruct s as [y | sym | c]; [ right | right | left ].
    - exact (any_usage_use _ _ _ Hc _ _ Hx (Huse _ (free_names_var _ _ Hin))).
    - exact (any_usage_use _ _ _ Hc _ _ Hx
               (Huse _ (free_names_symbol _ _ Hin))).
    - exists c. reflexivity. }
  destruct (rewrite_simples_au en vars vars' Henv args vs Hargs Hvs)
    as (vs' & Hvs' & Hall).
  assert (Hsims : Forall2 sim vs vs')
    by exact (Forall2_impl _ element_rel_sim Hall).
  destruct (eval_prim_related sim statics statics' p vs vs' v
              (sim_respects statics statics' Hst) Hsims Hev)
    as (v' & Hev' & Hsim).
  exists v'. cbn [eval_named]. rewrite Hvs'. split; [ exact Hev' | ].
  pose proof (eval_simples_length _ _ _ Hvs) as Hlen.
  assert (Hlen' : length vs' = length args)
    by (rewrite <- (Forall2_length Hall); exact Hlen).
  destruct (default_prim_cases statics' p args vs' v' Hcl Hlen' Hev')
    as [(c' & ->) | Ht].
  { rewrite (sim_const_r _ _ Hsim). cbn [rel].
    split; [ intros _; exact Hasx | reflexivity ]. }
  destruct (default_prim_cases statics p args vs v Hcl Hlen Hev)
    as [(c & ->) | [(k & -> & ->) | [(ak & lk & mut & arr & idx & elements
                                       & -> & -> & Harr & Hload)
                                     | (ak & -> & ->)]]].
  - cbn [sim] in Hsim. subst v'. cbn [rel].
    split; [ intros _; exact Hasx | reflexivity ].
  - destruct Ht as [(k' & _ & ->) | [(ak' & lk' & mut' & arr' & idx'
                                     & elements' & Hp & _) | (ak' & Hp & _)]];
      [ | discriminate Hp | discriminate Hp ].
    inversion Hall as [ | w w' ws ws' Hw _ ]; subst.
    exact (element_rel_at _ _ x Hw Hasx).
  - destruct Ht as [(k' & Hp & _) | [(ak' & lk' & mut' & arr' & idx'
                                     & elements' & Hp & -> & Harr' & Hload')
                                   | (ak' & Hp & _)]];
      [ discriminate Hp | | discriminate Hp ].
    injection Hp as <- <- <-.
    inversion Hall as [ | ? ? ? ? Harr_rel Hrest ]; subst.
    inversion Hrest as [ | ? ? ? ? Hidx_rel _ ]; subst.
    assert (Hidx : idx' = idx).
    { pose proof (element_rel_sim _ _ Hidx_rel) as Hs.
      destruct idx as [ci | | | | | | | | ];
        try (unfold array_loaded, array_index in Hload; simpl in Hload;
             discriminate Hload).
      exact Hs. }
    subst idx'.
    destruct arr as [ | | | s | | els | | | ]; simpl in Harr;
      try discriminate Harr.
    + destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
        [ | discriminate Harr ].
      destruct (proj1 (proj2 (proj2 Hst)) s _ Hl)
        as [(t & f & ->) | [(sl & sy & de & vs0 & -> & _) | (Ho & Hbound)]];
        [ discriminate Harr | discriminate Harr | ].
      destruct allocated as [ | | | | | els | ]; try discriminate Harr.
      injection Harr as <-.
      destruct (element_rel_sim _ _ Harr_rel) as [-> Hau_s].
      destruct Ho as [Hn | (els' & Hs' & Hf)];
        [ destruct (Hbound (au_hu _ _ _ Hc _ Hau_s) Hn) | ].
      simpl in Harr'. rewrite Hs' in Harr'. injection Harr' as <-.
      exact (element_rel_at _ _ x
               (array_loaded_related _ lk _ _ _ _ _
                  (Forall2_values_related _ _ (Hf Hau_s)) Hload Hload')
               Hasx).
    + injection Harr as <-.
      pose proof (element_rel_at _ _ x Harr_rel Hasx) as Hr.
      cbn [rel] in Hr. destruct Hr as (_ & elements'' & -> & Hels).
      simpl in Harr'. injection Harr' as <-.
      exact (element_rel_at _ _ x
               (array_loaded_related _ lk _ _ _ _ _ Hels Hload Hload') Hasx).
  - destruct Ht as [(k' & Hp & _) | [(ak' & lk' & mut' & arr' & idx'
                                     & elements' & Hp & _) | (ak' & _ & ->)]];
      [ discriminate Hp | discriminate Hp | ].
    cbn [rel]. split; [ exact Hasx | ]. exists vs'.
    split; [ reflexivity | exact (Forall2_values_related _ _ Hall) ].
Qed.

(* A [Let] of kind [Region] or [Rec_info], which the rebuild keeps: its
   value is a constant, of a simple or of the identity, the same in the
   two runs. *)
Lemma region_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (d : named) (v : value)
    {ctx : option code_id} :
  env_ok (env_kinds en) vars vars' ->
  named_closure_free d = true -> named_edges_ ctx x d ->
  eval_named statics vars d = Some v ->
  kind_of_value v = Region ∨ kind_of_value v = Rec_info ->
  flows_ v (Node_variable x) ->
  ∃ v' : value,
    eval_named statics' vars' (rebuild_named sol en x d) = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hfrag Hedges Hev Hk Hflows.
  destruct (not_value_const v (region_not_value _ Hk)) as (c & ->).
  exists (Value_const c).
  split; [ | split; [ intros _; exact Hflows | reflexivity ] ].
  destruct d as [s | p args | | | | | ]; simpl in Hfrag; try discriminate Hfrag;
    simpl in Hedges.
  - destruct Hedges as [_ Hnoted]. cbn [eval_named rebuild_named] in Hev |- *.
    destruct (rewrite_simple_eval en vars vars' s _ Henv Hnoted Hev)
      as (v' & Hs' & _ & Hkept).
    destruct (Hkept (kept_of_region en vars vars' s _ Henv Hev Hk))
      as [_ [_ ->]].
    exact Hs'.
  - cbn [eval_named] in Hev.
    destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    destruct (eval_prim_kinds _ _ _ _ Hev) as [_ Hres].
    pose proof Hk as Hkp. rewrite Hres in Hkp.
    destruct (region_prim p Hkp) as [(k & ->) | (r & ->)];
      [ | destruct (eval_prim_arity _ _ _ _ Hev)
            as [(? & ? & Hp & _) | [(? & ? & ? & Hp & _) | (? & Hp & _)]];
          discriminate Hp ].
    destruct (eval_prim_arity _ _ _ _ Hev)
      as [(up & v1 & Hp & -> & Hu) | [(bp & v1 & v2 & Hp & _) | (vp & Hp & _)]];
      [ | discriminate Hp | discriminate Hp ].
    injection Hp as <-.
    destruct args as [ | s1 [ | s2 args]]; simpl in Hvs;
      try (destruct (eval_simple vars s1); [ | discriminate Hvs ];
           destruct (eval_simple vars s2); [ | discriminate Hvs ];
           destruct (eval_simples vars args); discriminate Hvs);
      [ discriminate Hvs | ].
    destruct (eval_simple vars s1) as [w | ] eqn:Hs1; [ | discriminate Hvs ].
    injection Hvs as <-.
    cbn [eval_unary_primitive] in Hu.
    destruct (kind_eqb (kind_of_value w) k && bound_if_symbol statics w)
      eqn:Hcond; [ | discriminate Hu ].
    injection Hu as ->.
    assert (Hnoted : simple_noted noted s1).
    { destruct s1; simpl; try exact I.
      simpl in Hs1. injection Hs1 as Heq. discriminate Heq. }
    destruct (rewrite_simple_eval en vars vars' s1 _ Henv Hnoted Hs1)
      as (w' & Hs1' & _ & Hkept).
    destruct (Hkept (kept_of_region en vars vars' s1 _ Henv Hs1 Hk))
      as [Hr [_ ->]].
    rewrite Hr in Hs1'. cbn [rebuild_named].
    rewrite (unary_target en vars' statics' (Opaque_identity k) s1 _ Hr Hs1').
    cbn [eval_unary_primitive].
    apply andb_true_iff in Hcond as [Hkind _]. rewrite Hkind. reflexivity.
Qed.

(* The projection of a function slot from a closure of a set, related at
   the variable that the projection binds. *)
Lemma rel_project_function_slot (slot o : function_slot)
    (decls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)) (base y : node) (id0 id1 : code_id) :
  is_local (Field_function_slot o) = false ->
  lookup slot decls = Some id0 -> lookup o decls = Some id1 ->
  g_accessor G y (Field_function_slot o) base ->
  rel (Value_closure_of_set slot decls vs) base
    (Value_closure_of_set slot decls vs') ->
  rel (Value_closure_of_set o decls vs) y (Value_closure_of_set o decls vs').
Proof.
  intros Hl Hl0 Hl1 Hacc Hrel. cbn [rel] in Hrel.
  destruct Hrel as (sites & vs'' & Hmay & Heq & Hex & Hall).
  injection Heq as <-.
  destruct (Hall slot id0 Hl0) as (_ & _ & Hfs & _).
  apply (rel_read G sol keep called (exact = true) True is_local Hc _ _ _ _ _
      _ _ Hl
           (Hfs o id1 Hl1) Hacc Hmay).
  cbn [rel]. exists sites, vs'.
  split; [ exact (proj1 (Hall o id1 Hl1)) | ].
  split; [ reflexivity | split; [ exact Hex | exact Hall ] ].
Qed.

(* The value of a [Let] of a projection, that the rebuild keeps: a
   function slot of a closure, of the same set in the two runs; a value
   slot used, kept by the rebuild. *)
Lemma closure_named_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (d : named) (v : value)
    {ctx : option code_id} :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' -> env_flows_ vars -> statics_flow_ statics ->
  named_no_code (kind_of_value v) d = true -> named_plain d = true ->
  named_closure_free d = false ->
  named_edges_ ctx x d ->
  eval_named statics vars d = Some v ->
  is_var_used sol x (kind_of_value v) = true ->
  ∃ v' : value,
    eval_named statics' vars' (rebuild_named sol en x d) = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst Hvars Hsflow Hnc Hpl Hcf Hedges Hev Hused.
  destruct d as [s | p args | code vs | slot decls vs | w c | pf w c
                | mf mt c];
    simpl in Hpl, Hcf; try discriminate Hpl; try discriminate Hcf;
    [ simpl in Hnc; discriminate Hnc | | ].
  - (* [Project_value_slot_from] *)
    assert (Hx : hu (Node_variable x)).
    { simpl in Hnc. destruct (kind_of_value v); try discriminate Hnc;
        exact Hused. }
    simpl in Hedges. destruct Hedges as ((Hacc & Hnoted) & Hk & Hslots).
    pose proof (Hslots (Field_value_slot w)) as Hl.
    cbn [eval_named rebuild_named] in Hev |- *.
    destruct (eval_simple vars c) as [vc | ] eqn:Hvc; [ | discriminate Hev ].
    pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc) as Hc_hu.
    destruct (rewrite_simple_eval en vars vars' c vc Henv Hnoted Hvc)
      as (vc' & Hs' & _ & Hkept).
    destruct (Hkept (kept_of_hu en c Hc_hu)) as [Hr Hrel].
    rewrite Hr in Hs' |- *. rewrite Hs'. cbv beta iota.
    apply project_value_slot_defined in Hev as (decls & slots & Hcis & Hw).
    destruct vc as [ | code slots0 | slot0 decls0 slots0 | s | | | | | ];
      simpl in Hcis; try discriminate Hcis.
    + cbn [rel] in Hrel. destruct Hrel.
    + destruct (lookup slot0 decls0) as [id0 | ] eqn:Hl0;
        [ | discriminate Hcis ].
      injection Hcis as -> -> ->.
      cbn [rel] in Hrel. destruct Hrel as (sites & vs' & Hmay & -> & _ & Hall).
      destruct (Hall pf id0 Hl0) as (_ & _ & _ & Hsr).
      destruct (proj1 (slots_related_iff _ _) Hsr w v Hw Hk)
        as (from & Hcons & Hfu).
      pose proof (field_used_read is_local G sol Hc _ _ _ _ _ Hl Hcons Hacc
                    Hmay Hx) as Hused_w.
      destruct (Hfu Hused_w) as (x' & Hx' & Hrelx).
      pose proof (field_used_hu G sol is_local Hc Hexact _ _ _ Hl Hused_w Hcons)
        as Hhu_from.
      exists x'. split.
      * unfold project_value_slot. simpl. rewrite Hl0. simpl.
        rewrite Pos.eqb_refl. exact Hx'.
      * exact (rel_read G sol keep called (exact = true) True is_local Hc _ _
          _ _ _
        _ _ Hl
                 Hcons Hacc Hmay (Hrelx Hhu_from)).
    + destruct (lookup s (static_values statics))
        as [[ | slot0 syms0 decls0 slots0 | | | | | ] | ] eqn:Hls;
        try discriminate Hcis.
      * destruct (Hsflow s _ Hls) as [_ []].
      * destruct (lookup slot0 decls0) as [id0 | ] eqn:Hl0;
          [ | discriminate Hcis ].
        injection Hcis as -> -> ->.
        cbn [rel] in Hrel. destruct Hrel as (-> & Hmay & Hhu_s).
        destruct (proj1 (proj2 (proj2 Hst)) s _ Hls)
          as [(t & f & Heq)
             | [(sl & sy & de & vs0 & Heq & Htarget & Hbound) | ([] & _)]];
          [ discriminate Heq | ].
        injection Heq as <- <- <- <-.
        destruct Htarget as [Hnone | (vs' & Hls' & _ & Hsr)];
          [ destruct (Hbound (Hhu_s Hc_hu) Hnone) | ].
        destruct (proj1 (slots_related_iff _ _) Hsr w v Hw Hk)
          as (from & Hcons & Hfu).
        pose proof (field_used_read is_local G sol Hc _ _ _ _ _ Hl Hcons Hacc
                      Hmay Hx) as Hused_w.
        destruct (Hfu Hused_w) as (x' & Hx' & Hrelx).
        pose proof (field_used_hu G sol is_local Hc Hexact _ _ _ Hl Hused_w
                      Hcons) as Hhu_from.
        exists x'. split.
        -- unfold project_value_slot. simpl. rewrite Hls'. simpl.
           rewrite Hl0. simpl. rewrite Pos.eqb_refl. exact Hx'.
        -- exact (rel_read G sol keep called (exact = true) True is_local Hc _
            _ _
          _ _ _ _
                   Hl Hcons Hacc Hmay (Hrelx Hhu_from)).
  - (* [Project_function_slot] *)
    destruct Hedges as ((Hacc & Hnoted) & Hslots).
    cbn [eval_named rebuild_named] in Hev |- *.
    destruct (eval_simple vars c) as [vc | ] eqn:Hvc; [ | discriminate Hev ].
    pose proof (simple_flows is_local G sol noted cds keep no_loc Hc Hconsts
        Hsyms _ _
                  _ Hvars Hnoted Hvc) as Hfc.
    destruct (rewrite_simple_eval en vars vars' c vc Henv Hnoted Hvc)
      as (vc' & Hs' & _ & Hkept).
    unfold project_function_slot in Hev.
    destruct vc as [ | code slots0 | slot0 decls0 slots0 | s | | | | | ];
      simpl in Hev; try discriminate Hev.
    + destruct Hfc.
    + destruct (Pos.eqb_spec slot0 mf) as [-> | _]; [ | discriminate Hev ].
      destruct (lookup mf decls0) as [id0 | ] eqn:Hl0; [ | discriminate Hev ].
      destruct (lookup mt decls0) as [id1 | ] eqn:Hl1; [ | discriminate Hev ].
      injection Hev as <-.
      assert (Hx : hu (Node_variable x)) by exact Hused.
      destruct (Hkept (kept_of_hu en c
                         (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc)))
        as [Hr Hrel].
      rewrite Hr in Hs' |- *.
      pose proof Hrel as Hrel0. cbn [rel] in Hrel0.
      destruct Hrel0 as (sites & vs' & _ & -> & _).
      rewrite Hs'. eexists. split.
      * unfold project_function_slot. simpl.
        rewrite Pos.eqb_refl, Hl0, Hl1. reflexivity.
      * exact (rel_project_function_slot mf mt decls0 slots0 vs' _ _ id0 id1
                 (Hslots _) Hl0 Hl1 Hacc Hrel).
    + destruct (lookup s (static_values statics))
        as [[ | slot0 syms0 decls0 slots0 | | | | | ] | ] eqn:Hls;
        try discriminate Hev.
      * destruct (Hsflow s _ Hls) as [_ []].
      * destruct (Pos.eqb_spec slot0 mf) as [-> | _]; [ | discriminate Hev ].
        destruct (lookup mf decls0) as [id0 | ] eqn:Hl0;
          [ | discriminate Hev ].
        destruct (lookup mt decls0) as [id1 | ] eqn:Hl1;
          [ | discriminate Hev ].
        destruct (lookup mt syms0) as [s' | ] eqn:Hs'0;
          [ | discriminate Hev ].
        injection Hev as <-.
        assert (Hx : hu (Node_variable x)) by exact Hused.
        pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc) as Hc_hu.
        destruct (Hkept (kept_of_hu en c Hc_hu)) as [Hr Hrel].
        rewrite Hr in Hs' |- *. cbn [rel] in Hrel.
        destruct Hrel as (-> & Hmay & Hhu).
        destruct (proj1 (proj2 (proj2 Hst)) s _ Hls)
          as [(tag & fields & Heq) | [(sl & sy & de & vs0 & Heq & Htarget
                                      & Hbound) | ([] & _)]];
          [ discriminate Heq | ].
        injection Heq as <- <- <- <-.
        destruct Htarget as [Hnone | (vs' & Hls' & _)];
          [ destruct (Hbound (Hhu Hc_hu) Hnone) | ].
        destruct (Hsflow s _ Hls) as (_ & _ & Hfs & _).
        destruct (Hfs mt s' Hs'0) as [Hcons Hself].
        rewrite Hs'. eexists. split.
        -- unfold project_function_slot. simpl. rewrite Hls'. simpl.
           rewrite Pos.eqb_refl, Hl0, Hl1, Hs'0. reflexivity.
        -- apply (rel_read G sol keep called (exact = true) True is_local Hc _
            _ _
          _ _ _ _
                    (Hslots _) Hcons Hacc Hmay).
           cbn [rel]. split; [ reflexivity | split; [ exact Hself | ] ].
           exact (fun h => h).
Qed.

(* The value of a [Let] that the rebuild keeps, in the second run. *)
Lemma named_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (d : named) (v : value)
    {ctx : option code_id} :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' -> env_flows_ vars -> statics_flow_ statics ->
  named_checked exact (kind_of_value v) d = true -> named_plain d = true ->
  named_edges_ ctx x d ->
  eval_named statics vars d = Some v ->
  flows_ v (Node_variable x) ->
  is_var_used sol x (kind_of_value v) = true ->
  ∃ v' : value,
    eval_named statics' vars' (rebuild_named sol en x d) = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst Hvars Hsflow Hnc Hpl Hedges Hev Hflows Hused.
  destruct (named_closure_free d) eqn:Hfrag;
    [ | exact (closure_named_rel en vars vars' statics statics' x d v Henv
                 Hst Hvars Hsflow (named_checked_no_code _ _ _ Hnc) Hpl Hfrag
                 Hedges Hev Hused) ].
  destruct (is_var_used_cases x _ Hused) as [Hx | Hk];
    [ | exact (region_rel en vars vars' statics statics' x d v Henv Hfrag
                 Hedges Hev Hk Hflows) ].
  destruct d as [s | p args | | | | | ]; simpl in Hfrag; try discriminate Hfrag;
    simpl in Hedges.
  - (* [Simple] *)
    destruct Hedges as [Hal Hnoted]. cbn [eval_named rebuild_named] in Hev |- *.
    destruct (rewrite_simple_eval en vars vars' s v Henv Hnoted Hev)
      as (v' & Hs' & _ & Hkept).
    pose proof (alias_input _ _ _ Hc _ _ Hal) as Hal'.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hx Hal') as Hs_hu.
    destruct (Hkept (kept_of_hu en s Hs_hu)) as [_ Hrel].
    exists v'. split; [ exact Hs' | ].
    exact (rel_alias G sol keep called (exact = true) True is_local Hc _ _ _ _
        Hal'
      Hrel).
  - (* [Prim] *)
    cbn [eval_named] in Hev.
    destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    unfold prim_edges in Hedges. destruct Hedges as [_ Hedges].
    destruct (prim_classified p args) eqn:Hcl.
    + destruct p as [up | bp | vp | hp | rp].
      * destruct up as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                       | | bn | bn | | | r];
          destruct args as [ | s1 [ | s2 args]]; simpl in Hcl;
          try discriminate Hcl; simpl in Hedges;
          destruct Hedges as [Hedge Hnoted]; simpl in Hvs;
          destruct (eval_simple vars s1) as [v1 | ] eqn:Hs1;
          try discriminate Hvs; injection Hvs as <-;
          rewrite eval_prim_unary in Hev;
          [ | | | | exact (box_number_rel en vars vars' statics statics' x bn
                             s1 v1 v Henv Hvars Hedge Hnoted Hs1 Hev) ];
          pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hedge) as Hs1_hu;
          destruct (rewrite_simple_eval en vars vars' s1 v1 Henv Hnoted Hs1)
            as (v1' & Hs1' & _ & Hkept);
          destruct (Hkept (kept_of_hu en s1 Hs1_hu)) as [Hr Hrel];
          rewrite Hr in Hs1'; cbn [rebuild_named];
          rewrite (unary_target en vars' statics' _ s1 v1' Hr Hs1').
        -- (* [Block_load] *)
           exact (block_load_rel statics statics' x _ kind fld v1 v1' v Hst
                    Hedge Hrel Hx Hev).
        -- (* [Is_int] *)
           cbn [eval_unary_primitive] in Hev |- *.
           rewrite (is_int_rel statics statics' v1 v1' _ Hst Hrel Hs1_hu).
           exists v. split; [ exact Hev | ].
           destruct (eval_is_int statics true v1); [ | discriminate Hev ].
           injection Hev as <-.
           split; [ intros _; exact Hflows | reflexivity ].
        -- (* [Get_tag] *)
           exists v.
           split; [ exact (get_tag_rel statics statics' v1 v1' v _ Hst Hrel
                             Hs1_hu Hev) | ].
           cbn [eval_unary_primitive] in Hev.
           destruct (block_of statics v1) as [[tag fields] | ];
             [ | discriminate Hev ].
           injection Hev as <-.
           split; [ intros _; exact Hflows | reflexivity ].
        -- (* [Unbox_number] *)
           exists v.
           split; [ exact (unbox_rel statics statics' x _ bn v1 v1' v Hst
                             Hedge Hrel Hx Hev) | ].
           cbn [eval_unary_primitive] in Hev.
           destruct (boxed_number_of statics v1) as [c | ];
             [ | discriminate Hev ].
           destruct (is_number_of_kind bn c); [ | discriminate Hev ].
           injection Hev as <-.
           split; [ intros _; exact Hflows | reflexivity ].
      * discriminate Hcl.
      * destruct vp as [[tag] | ak]; [ | discriminate Hcl ].
        rewrite eval_prim_variadic in Hev. cbn [rebuild_named].
        exact (make_block_rel en vars vars' statics statics' x tag args vs v
                 Henv Hst Hedges Hvs Hev).
      * destruct (eval_prim_arity _ _ _ _ Hev)
          as [(up & v1 & Hp & _) | [(bp & v1 & v2 & Hp & _) | (vp & Hp & _)]];
          discriminate Hp.
      * destruct (eval_prim_arity _ _ _ _ Hev)
          as [(up & v1 & Hp & _) | [(bp & v1 & v2 & Hp & _) | (vp & Hp & _)]];
          discriminate Hp.
    + assert (Hplain : rebuild_named sol en x (Prim p args)
                       = Prim p (map (rewrite_simple sol en) args)).
      { destruct (eval_prim_arity _ _ _ _ Hev)
          as [(up & v1 & -> & -> & _) | [(bp & v1 & v2 & -> & -> & _)
                                       | (vp & -> & _)]].
        - destruct up; try reflexivity.
          pose proof (eval_simples_length _ _ _ Hvs) as Hlen.
          destruct args as [ | s1 [ | s2 args]]; simpl in Hlen;
            try discriminate Hlen.
          discriminate Hcl.
        - reflexivity.
        - destruct vp as [[tag] | ak]; [ discriminate Hcl | reflexivity ]. }
      rewrite Hplain.
      exact (default_rel en vars vars' statics statics' x p args vs v Henv
               Hst Hedges Hcl Hx Hvs Hev).
Qed.

(* ------------------------------------------------------------------------ *)
(* Static bindings                                                          *)
(* ------------------------------------------------------------------------ *)

Lemma rebuild_binding_block (en : rebuild_env) (s : symbol) (tag : Z)
    (fields : list simple) :
  rebuild_b en (Bind_symbol s (Static_const_block tag fields))
  = if is_symbol_used sol s
    then [Bind_symbol s
            (Static_const_block tag
               (rebuild_fields sol en (Node_symbol s) 0 fields))]
    else [].
Proof. reflexivity. Qed.

(* A symbol of a static set has a use only if the set is kept. *)
Lemma lookup_existsb (f : symbol -> bool) (slot : function_slot)
    (closure_symbols : list (function_slot * symbol)) (s : symbol) :
  lookup slot closure_symbols = Some s -> f s = true ->
  existsb (fun '(_, sym) => f sym) closure_symbols = true.
Proof.
  induction closure_symbols as [ | [g sym] closure_symbols IH]; simpl;
    [ discriminate | ].
  destruct (Pos.eqb slot g).
  - intros Heq Hf. injection Heq as ->. rewrite Hf. reflexivity.
  - intros Hl Hf. rewrite (IH Hl Hf), orb_true_r. reflexivity.
Qed.

(* A simple of the first run has a value in the second, rewritten. *)
Lemma rewrite_simple_some (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_ok (env_kinds en) vars vars' ->
  eval_simple vars s = Some v ->
  ∃ v' : value, eval_simple vars' (rewrite_simple sol en s) = Some v'.
Proof.
  intros Henv Hs. destruct s as [x | sym | c]; simpl in Hs.
  - destruct (Henv x v Hs) as [_ Hused].
    unfold rewrite_simple. cbv zeta.
    rewrite (env_ok_kind en vars vars' x v Henv Hs).
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hu.
    + destruct (Hused eq_refl) as (v' & Hl & _). exists v'. exact Hl.
    + eexists. reflexivity.
  - unfold rewrite_simple. destruct (is_symbol_used sol sym); eexists;
      reflexivity.
  - eexists. reflexivity.
Qed.

Lemma lookup_rebuild_value_slots (en : rebuild_env) (bound : list node)
    (value_slots : list (value_slot * simple)) (w : value_slot) (s : simple) :
  lookup w value_slots = Some s ->
  existsb (fun n => s_field_used sol n (Field_value_slot w)) bound = true ->
  lookup w (rebuild_value_slots sol en bound value_slots)
  = Some (rewrite_simple sol en s).
Proof.
  intros Hl Hb. induction value_slots as [ | [w' s'] value_slots IH];
    simpl in Hl |- *; [ discriminate Hl | ].
  destruct (Pos.eqb_spec w w') as [-> | Hne].
  - injection Hl as ->. unfold field_used. rewrite Hb. simpl.
    rewrite Pos.eqb_refl. reflexivity.
  - unfold field_used.
    destruct (existsb (fun n => s_field_used sol n (Field_value_slot w'))
                bound); [ | exact (IH Hl) ].
    simpl. rewrite (proj2 (Pos.eqb_neq w w') Hne). exact (IH Hl).
Qed.

Lemma eval_value_slots_lookup' (vars : env)
    (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) (w : value_slot) (s : simple) :
  eval_value_slots vars value_slots = Some vs ->
  lookup w value_slots = Some s ->
  ∃ v : value, lookup w vs = Some v ∧ eval_simple vars s = Some v.
Proof.
  revert vs. induction value_slots as [ | [w' s'] value_slots IH];
    intros vs Hev Hl; simpl in Hev, Hl; [ discriminate Hl | ].
  destruct (eval_simple vars s') as [v' | ] eqn:Hs'; [ | discriminate Hev ].
  destruct (eval_value_slots vars value_slots) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hev ].
  injection Hev as <-. simpl.
  destruct (Pos.eqb w w').
  - injection Hl as ->. exists v'. auto.
  - exact (IH vs0 eq_refl Hl).
Qed.

(* The value slots of a closure in the second run: those that the rebuild
   keeps ([rebuild_value_slots]), related at a node of [bound] that has
   them used. *)
Lemma value_slots_rel (en : rebuild_env) (vars vars' : env)
    (bound : list node) (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) :
  env_ok (env_kinds en) vars vars' ->
  eval_value_slots vars value_slots = Some vs ->
  ∃ vs' : list (value_slot * value),
    eval_value_slots vars' (rebuild_value_slots sol en bound value_slots)
    = Some vs'
    ∧ (value_slots = [] -> vs' = vs)
    ∧ ∀ a : node,
        a ∈ bound ->
        (∀ (w : value_slot) (s : simple),
           lookup w value_slots = Some s -> keep w = true ->
           g_constructor G a (Field_value_slot w) (node_of_simple s)
           ∧ simple_noted noted s) ->
        slots_related (slot_rel a vs') vs [].
Proof.
  intros Henv Hev.
  assert (Hex : ∃ vs' : list (value_slot * value),
            eval_value_slots vars' (rebuild_value_slots sol en bound
                                      value_slots)
            = Some vs').
  { clear - Henv Hev. revert vs Hev.
    induction value_slots as [ | [w s] value_slots IH]; intros vs Hev;
      [ exists []; reflexivity | ].
    simpl in Hev.
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate Hev ].
    destruct (eval_value_slots vars value_slots) as [vs0 | ] eqn:Hvs0;
      [ | discriminate Hev ].
    destruct (IH vs0 eq_refl) as (vs0' & Hvs0').
    unfold rebuild_value_slots in Hvs0' |- *. cbn [flat_map].
    destruct (existsb _ bound); [ | exact (ex_intro _ vs0' Hvs0') ].
    destruct (rewrite_simple_some en vars vars' s v Henv Hs) as (v' & Hs').
    cbn [app eval_value_slots]. rewrite Hs', Hvs0'. eexists. reflexivity. }
  destruct Hex as (vs' & Hvs'). exists vs'.
  split; [ exact Hvs' | split ].
  - intros ->. simpl in Hev, Hvs'. injection Hev as <-.
    injection Hvs' as <-. reflexivity.
  - intros a Ha Hcons. apply slots_related_iff. intros w x Hx Hk.
    destruct (eval_value_slots_lookup _ _ _ _ _ Hev Hx) as (s & Hs & Hsx).
    destruct (Hcons w s Hs Hk) as [Hc_ Hnoted].
    exists (node_of_simple s). split; [ exact Hc_ | intros Hused ].
    assert (Hb : existsb (fun n => s_field_used sol n (Field_value_slot w))
                   bound = true).
    { apply existsb_exists. exists a. auto. }
    destruct (eval_value_slots_lookup' _ _ _ _ _ Hvs'
                (lookup_rebuild_value_slots en bound value_slots w s Hs Hb))
      as (x' & Hx' & Hsx').
    exists x'. split; [ exact Hx' | intros Hhu ].
    destruct (rewrite_simple_eval en vars vars' s x Henv Hnoted Hsx)
      as (x'' & Hs'' & _ & Hkept).
    destruct (Hkept (kept_of_hu en s Hhu)) as [_ Hrel].
    rewrite Hsx' in Hs''. injection Hs'' as <-. exact Hrel.
Qed.

(* The bindings of a [Let_static] of the fragment: those of the symbols
   with a use are kept, with their fields used related; the closures of a
   static set are kept if a symbol of the set has a use, with the value
   slots that the rebuild keeps. *)
Lemma static_bindings_rel (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (bindings : list static_binding) (statics statics' statics2 : static_env),
    forallb (binding_checked exact codes) bindings = true ->
    every binding_edges_ bindings ->
    statics_rel statics statics' ->
    eval_static_bindings vars bindings statics = Some statics2 ->
    ∃ statics2' : static_env,
      eval_static_bindings vars' (flat_map (rebuild_b en) bindings) statics'
      = Some statics2'
      ∧ statics_rel statics2 statics2'.
Proof.
  intros Henv. induction bindings as [ | b bindings IH];
    intros statics statics' statics2 Hfrag Hedges Hst Hev; simpl in Hev.
  - injection Hev as <-. exists statics'. split; [ reflexivity | exact Hst ].
  - destruct Hedges as [Hb Hbindings]. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hfrag_b Hfrag].
    destruct (binding_checked_cases _ _ _ Hfrag_b)
      as [Hnc | (id & c & -> & _ & _)].
    2: { simpl in Hev.
      destruct (lookup id (static_code statics)) eqn:Hl; [ discriminate Hev | ].
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      assert (Hl' : lookup id (static_code statics') = None).
      { destruct (Hcode id) as [H | [H _]]; [ | exact H ].
        rewrite H, Hl. reflexivity. }
      cbn [flat_map rebuild_binding_cc]. unfold is_code_id_used.
      destruct (s_has_usage sol (Node_code_id id)) eqn:Hu.
      - cbn [app eval_static_bindings]. rewrite Hl'.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ exact Hnone | split; [ exact Hblocks | split ] ];
          [ exact Honly | ].
        intros id'. simpl.
        destruct (Pos.eqb_spec id' id) as [-> | _];
          [ left; reflexivity | exact (Hcode id') ].
      - cbn [app eval_static_bindings].
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ exact Hnone | split; [ exact Hblocks | split ] ];
          [ exact Honly | ].
        intros id'. simpl.
        destruct (Pos.eqb_spec id' id) as [-> | _]; [ | exact (Hcode id') ].
        right. split; [ exact Hl' | ]. intros H. rewrite Hu in H.
        discriminate H. }
    destruct b as [id c | id | s [ | slot syms decls value_slots | tag fields
                                 | contents | bytes | elements ]];
      simpl in Hnc; try discriminate Hnc; simpl in Hb.
    + exact (IH _ _ _ Hfrag Hbindings Hst Hev).
    + apply andb_true_iff in Hnc as [_ Hself].
      destruct (lookup slot syms) as [s0 | ] eqn:Hss; [ | discriminate Hself ].
      apply Pos.eqb_eq in Hself. subst s0.
      destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev.
      destruct (lookup slot decls) as [id0 | ] eqn:Hid; [ | discriminate Hev ].
      destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
        [ | discriminate Hev ].
      cbv beta iota in Hev.
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      destruct Hb as (_ & _ & _ & Hslots).
      cbn [flat_map rebuild_binding_cc static_const_is_used].
      destruct (existsb (fun '(_, sym) => is_symbol_used sol sym) syms)
        eqn:Hused.
      * set (bound := map (fun '(_, sym) => Node_symbol sym) syms).
        destruct (value_slots_rel en vars vars' bound value_slots vs Henv Hvs)
          as (vs' & Hvs' & Hempty & Hrel).
        assert (Hs_in : Node_symbol s ∈ bound).
        { apply (in_map (fun '(_, sym) => Node_symbol sym) syms (slot, s)).
          exact (lookup_in _ _ _ Hss). }
        cbn [app eval_static_bindings rebuild_static_const].
        rewrite (Hnone s Hl).
        cbn [eval_static_const]. rewrite Hid. fold bound. rewrite Hvs'.
        cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. left.
           exists slot, syms, decls, vs. split; [ reflexivity | ].
           split; [ | intros _; discriminate ].
           right. exists vs'. split; [ reflexivity | split ].
           ++ intros He. cbn [binding_checked] in Hfrag_b.
              rewrite He in Hfrag_b. simpl in Hfrag_b.
              destruct value_slots;
                [ exact (Hempty eq_refl) | discriminate Hfrag_b ].
           ++ exact (Hrel _ Hs_in Hslots).
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. left.
           exists slot, syms, decls, vs. split; [ reflexivity | ].
           split; [ left; exact (Hnone s Hl) | intros Hu' ].
           rewrite (lookup_existsb (is_symbol_used sol) slot syms s Hss Hu')
             in Hused.
           discriminate Hused.
    + destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev.
      destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
        [ | discriminate Hev ].
      cbv beta iota in Hev.
      destruct Hb as (Hcons & _ & _ & Hnoted).
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      cbn [flat_map]. rewrite rebuild_binding_block. unfold is_symbol_used.
      destruct (s_has_usage sol (Node_symbol s)) eqn:Hu.
      * destruct (rebuild_fields_eval en vars vars' (Node_symbol s) Henv
                    fields 0 vs Hcons Hnoted Hvs)
          as (vs' & Hvs' & Hf & _).
        cbn [app eval_static_bindings]. rewrite (Hnone s Hl).
        cbn [eval_static_const]. rewrite Hvs'. cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Hblocks s' tag' fields' Hl' Hu') ].
           injection Hl' as <- <-. exists vs'.
           split; [ reflexivity | exact Hf ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ injection Hl' as <-; left; eexists _, _; reflexivity
             | exact (Honly s' allocated Hl') ].
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ congruence | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s);
             [ injection Hl' as <-; left; eexists _, _; reflexivity
             | exact (Honly s' allocated Hl') ].
    + (* A static boxed number. A variable in it has a use from the
         symbol, and is kept with it. *)
      destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev.
      destruct (eval_simple vars contents) as [v | ] eqn:Hv;
        [ | discriminate Hev ].
      cbv beta iota in Hev.
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      cbn [flat_map rebuild_binding_cc static_const_is_used
           rebuild_static_const].
      unfold is_symbol_used.
      destruct (s_has_usage sol (Node_symbol s)) eqn:Hu.
      * assert (Hcontents :
                  ∃ v' : value,
                    eval_simple vars' (rewrite_or_variable sol en contents)
                    = Some v'
                    ∧ (v' = v ∨ sim v v')).
        { destruct contents as [x | sym | c];
            [ | exists v; split; [ exact Hv | left; reflexivity ] .. ].
          pose proof (any_usage_use _ _ _ Hc _ _ Hu (proj2 Hb x eq_refl))
            as Hx.
          pose proof (au_hu _ _ _ Hc _ Hx) as Hhx.
          destruct (proj2 (Henv x v Hv) (is_var_used_hu x _ Hhx))
            as (v' & Hl' & Hrel).
          exists v'. split.
          - cbn [rewrite_or_variable]. rewrite (is_var_used_hu x _ Hhx).
            exact Hl'.
          - right. exact (rel_au_sim G sol keep called (exact = true) True
              is_local Hc Hlocal _ _ _ Hx Hrel). }
        destruct Hcontents as (v' & Hv' & Hvv).
        cbn [app eval_static_bindings]. rewrite (Hnone s Hl).
        cbn [eval_static_const]. rewrite Hv'. cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. right.
           split; [ right; exists v'; split; [ reflexivity | exact Hvv ]
                  | intros _; discriminate ].
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ congruence | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. right.
           split; [ left; exact (Hnone s Hl) | intros Hu'; congruence ].
    + (* A static string. *)
      destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev. cbv beta iota in Hev.
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      cbn [flat_map rebuild_binding_cc static_const_is_used
           rebuild_static_const].
      unfold is_symbol_used.
      destruct (s_has_usage sol (Node_symbol s)) eqn:Hu.
      * cbn [app eval_static_bindings]. rewrite (Hnone s Hl).
        cbn [eval_static_const]. cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. right.
           split; [ right; reflexivity | intros _; discriminate ].
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ congruence | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. right.
           split; [ left; exact (Hnone s Hl) | intros Hu'; congruence ].
    + (* A static immutable array: its elements are those of a block, all
         used if its symbol has any usage. *)
      destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev.
      destruct (eval_simples vars elements) as [vs | ] eqn:Hvs;
        [ | discriminate Hev ].
      cbv beta iota in Hev.
      assert (Hcons : ∀ (i : nat) (e0 : simple),
                nth_error elements i = Some e0 ->
                g_constructor G (Node_symbol s) (Field_block i Value)
                  (node_of_simple e0))
        by (destruct elements;
            [ intros [ | i ] e0 H; discriminate H | exact (proj1 Hb) ]).
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      cbn [flat_map rebuild_binding_cc static_const_is_used
           rebuild_static_const].
      unfold is_symbol_used.
      destruct (s_has_usage sol (Node_symbol s)) eqn:Hu.
      * destruct (rewrite_simples_kinds en vars vars' Henv elements vs Hvs)
          as (vs' & Hvs' & _).
        assert (Hels : au (Node_symbol s) -> Forall2 element_rel vs vs').
        { intros Hau.
          assert (Hargs : ∀ e0 : simple, e0 ∈ elements ->
                    (∃ c : const, e0 = Const c) ∨ au (node_of_simple e0)).
          { intros e0 Hin. right.
            destruct (In_nth_error _ _ Hin) as (i & Hi).
            exact (any_usage_constructor _ _ _ Hc _ _ _ Hau (Hcons i e0 Hi)
                     (block_field_not_local is_local Hlocal i Value)). }
          destruct (rewrite_simples_au en vars vars' Henv elements vs Hargs
                      Hvs)
            as (vs'' & Hvs'' & Hall).
          rewrite Hvs' in Hvs''. injection Hvs'' as <-. exact Hall. }
        cbn [app eval_static_bindings]. rewrite (Hnone s Hl).
        cbn [eval_static_const]. rewrite Hvs'. cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. right.
           split; [ right; exists vs'; split; [ reflexivity | exact Hels ]
                  | intros _; discriminate ].
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ congruence | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right. right.
           split; [ left; exact (Hnone s Hl) | intros Hu'; congruence ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Parameters                                                               *)
(* ------------------------------------------------------------------------ *)

(* The parameters kept by a mask, bound in the second run to the values
   kept: each parameter that the rebuild keeps is bound to a related
   value. *)
Lemma bind_params_rel (params : list variable) :
  ∀ (vs vs' : list value) (mask : list bool)
    (kinds : finite_map flambda_kind) (vars vars' hv : env),
    env_ok kinds vars vars' ->
    length vs' = length vs ->
    length mask = length params ->
    (∀ (i : nat) (p : variable) (v v' : value),
       nth_error params i = Some p -> nth_error vs i = Some v ->
       nth_error vs' i = Some v' ->
       is_var_used sol p (kind_of_value v) = true ->
       nth_error mask i = Some true ∧ rel v (Node_variable p) v') ->
    bind_params params vs vars = Some hv ->
    ∃ hv' : env,
      bind_params (filter_kept mask params) (filter_kept mask vs') vars'
      = Some hv'
      ∧ env_ok (rev (combine params (map kind_of_value vs)) ++ kinds) hv hv'.
Proof.
  induction params as [ | p params IH];
    intros [ | v vs] [ | v' vs'] [ | m mask] kinds vars vars' hv Henv Hlen
      Hmlen Hrel Hbind;
    simpl in Hbind, Hlen, Hmlen; try discriminate.
  - injection Hbind as <-. exists vars'. split; [ reflexivity | exact Henv ].
  - assert (Hrel' : ∀ (i : nat) (p' : variable) (w w' : value),
              nth_error params i = Some p' -> nth_error vs i = Some w ->
              nth_error vs' i = Some w' ->
              is_var_used sol p' (kind_of_value w) = true ->
              nth_error mask i = Some true ∧ rel w (Node_variable p') w').
    { intros i. exact (Hrel (S i)). }
    assert (Henv' : env_ok ((p, kind_of_value v) :: kinds) ((p, v) :: vars)
                      (if m then (p, v') :: vars' else vars')).
    { apply (env_ok_bind kinds vars vars'); [ exact Henv | | ].
      - intros Hu. destruct (Hrel 0 p v v' eq_refl eq_refl eq_refl Hu)
          as [[= ->] Hr].
        exists v'. auto.
      - destruct m; [ right; exists v'; reflexivity | left; reflexivity ]. }
    destruct (IH vs vs' mask _ _ _ hv Henv' ltac:(lia) ltac:(lia) Hrel' Hbind)
      as (hv' & Hbind' & Hok).
    exists hv'. split.
    + destruct m; exact Hbind'.
    + simpl. rewrite <- app_assoc. exact Hok.
Qed.

Lemma with_kinds_combine (params : list variable) :
  ∀ kinds : list flambda_kind,
    length kinds = length params ->
    with_kinds params kinds = combine params kinds.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds] Hlen;
    simpl in Hlen |- *; try discriminate; [ reflexivity | ].
  f_equal. apply IH. lia.
Qed.

Lemma with_kinds_length (params : list variable) :
  ∀ kinds : list flambda_kind, length (with_kinds params kinds) = length params.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds]; simpl;
    try rewrite IH; reflexivity.
Qed.

Lemma with_kinds_nth (params : list variable) :
  ∀ (kinds : list flambda_kind) (i : nat) (p : variable) (k : flambda_kind),
    nth_error params i = Some p -> nth_error kinds i = Some k ->
    nth_error (with_kinds params kinds) i = Some (p, k).
Proof.
  induction params as [ | p0 params IH]; intros [ | k0 kinds] i p k Hp Hk;
    try (destruct i; discriminate).
  destruct i as [ | i]; simpl in Hp, Hk |- *.
  - injection Hp as <-. injection Hk as <-. reflexivity.
  - exact (IH kinds i p k Hp Hk).
Qed.

Lemma handler_mask_length (b : bool) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  length (handler_mask sol b (Handler params metadata handler))
  = length params.
Proof.
  unfold handler_mask, params_to_keep. rewrite length_map.
  apply with_kinds_length.
Qed.

(* A parameter that the rebuild keeps is kept by the mask. *)
Lemma handler_mask_used (b : bool) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (i : nat) (p : variable)
    (k : flambda_kind) :
  nth_error params i = Some p ->
  nth_error (hm_params_kinds metadata) i = Some k ->
  is_var_used sol p k = true ->
  nth_error (handler_mask sol b (Handler params metadata handler)) i
  = Some true.
Proof.
  intros Hp Hk Hu. unfold handler_mask, params_to_keep.
  rewrite nth_error_map, (with_kinds_nth _ _ _ _ _ Hp Hk). simpl.
  rewrite Hu. reflexivity.
Qed.

(* An exception handler keeps its first parameter. *)
Lemma handler_mask_exn_first (h : continuation_handler) :
  handler_is_exn_handler h = true ->
  hd true (handler_mask sol (handler_is_exn_handler h) h) = true.
Proof.
  intros He. rewrite He.
  destruct h as [[ | p params] metadata handler]; [ reflexivity | ].
  unfold handler_mask, params_to_keep.
  destruct (hm_params_kinds metadata) as [ | k0 kinds]; cbn;
    rewrite Pos.eqb_refl, orb_true_r; reflexivity.
Qed.

Lemma filter_kept_map {A B : Type} (f : A -> B) (mask : list bool) :
  ∀ l : list A, filter_kept mask (map f l) = map f (filter_kept mask l).
Proof.
  induction mask as [ | m mask IH]; intros [ | a l]; simpl; try reflexivity.
  - f_equal. clear a. induction l as [ | a l IHl]; simpl; [ reflexivity | ].
    f_equal. exact IHl.
  - destruct m; simpl; [ f_equal | ]; apply IH.
Qed.

Lemma eval_simples_filter_kept (vars : env) (mask : list bool) :
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    eval_simples vars (filter_kept mask args) = Some (filter_kept mask vs).
Proof.
  induction mask as [ | m mask IH]; intros args vs Hvs.
  - assert (Hnil : ∀ l : list simple, filter_kept [] l = l).
    { induction l as [ | a l IHl]; simpl;
        [ reflexivity | f_equal; exact IHl ]. }
    assert (Hnil' : ∀ l : list value, filter_kept [] l = l).
    { induction l as [ | a l IHl]; simpl;
        [ reflexivity | f_equal; exact IHl ]. }
    rewrite Hnil, Hnil'. exact Hvs.
  - destruct args as [ | s args]; simpl in Hvs.
    + injection Hvs as <-. reflexivity.
    + destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
      destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
      injection Hvs as <-. simpl. destruct m.
      * simpl. rewrite Hs, (IH args ws Hws). reflexivity.
      * exact (IH args ws Hws).
Qed.

Lemma with_params_kinds_kinds (metadata : handler_metadata)
    (kinds : list flambda_kind) :
  hm_params_kinds (with_params_kinds metadata kinds) = kinds.
Proof.
  unfold with_params_kinds. destruct (hm_is_exn_handler metadata); reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Lookups in a recursive group                                             *)
(* ------------------------------------------------------------------------ *)

Lemma lookup_map_pairs {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l)
  = match lookup k l with Some a => Some (f a) | None => None end.
Proof.
  induction l as [ | [k' a] l IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma lookup_map_pairs_app {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) (rest : list (positive * B)) :
  lookup k (map (fun '(k', a) => (k', f a)) l ++ rest)
  = match lookup k l with Some a => Some (f a) | None => lookup k rest end.
Proof.
  rewrite lookup_app, lookup_map_pairs. destruct (lookup k l); reflexivity.
Qed.

Lemma lookup_rec_closures
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) (k : continuation) :
  lookup k (rec_closures handlers vars conts ++ rest)
  = match lookup k handlers with
    | Some h => Some (Closure h vars conts handlers)
    | None => lookup k rest
    end.
Proof.
  unfold rec_closures.
  exact (lookup_map_pairs_app (fun h => Closure h vars conts handlers) k
           handlers rest).
Qed.

Lemma lookup_none_among (k : positive) {A : Type} (l : list (positive * A)) :
  lookup k l = None -> among k (map fst l) = false.
Proof.
  intros H. destruct (among k (map fst l)) eqn:Hin; [ | reflexivity ].
  apply among_In in Hin. destruct (lookup_none_not_in k l H Hin).
Qed.

(* ------------------------------------------------------------------------ *)
(* The rebuild, construct by construct                                      *)
(* ------------------------------------------------------------------------ *)

(* The cases of the rebuild of a [Let]. *)
Lemma rebuild_let_cases (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  rebuild en (Let x kind d body) = Invalid
  ∨ (∃ (d' : named) (en' : rebuild_env),
       rebuild en (Let x kind d body) = Let x kind d' (rebuild en' body))
  ∨ ∃ en' : rebuild_env, rebuild en (Let x kind d body) = rebuild en' body.
Proof.
  cbn [rebuild_expr_cc].
  destruct (is_pending_projection en x d); [ right; left; eauto | ].
  destruct d; cbv zeta;
    repeat match goal with
           | |- context [if ?b then _ else _] => destruct b
           end; eauto.
Qed.

(* A [Let] that is not of a set of closures, with no projection pending,
   or the next projection pending, of a function slot. *)

Lemma rebuild_let_pending_eq (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (move_from move_to : function_slot)
    (closure : simple) (body : expr) (ys : list variable) :
  env_projections en = x :: ys ->
  rebuild en (Let x kind (Project_function_slot move_from move_to closure)
                body)
  = Let x kind (Project_function_slot move_from move_to closure)
      (rebuild (bind_variables_then en [(x, kind)] ys) body).
Proof.
  intros Hep. cbn [rebuild_expr_cc]. unfold is_pending_projection.
  rewrite Hep, Pos.eqb_refl. reflexivity.
Qed.

(* A closure of a set: it is kept if it or a closure projected from it
   next has a use, and these are then pending. *)
Lemma rebuild_let_set_eq (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (body : expr) :
  env_projections en = [] ->
  rebuild en (Let x kind (Closure_of_set slot decls value_slots) body)
  = if is_dead_var sol x kind
       || existsb (fun y => is_dead_var sol y Value)
            (projected_closures x slot (other_slots slot decls) body)
    then Invalid
    else if is_var_used sol x kind
            || existsb (fun y => is_var_used sol y Value)
                 (projected_closures x slot (other_slots slot decls) body)
    then
      Let x kind
        (Closure_of_set slot decls
           (rebuild_value_slots sol en (set_closure_nodes F (Node_variable x))
              value_slots))
        (rebuild
           (bind_variables_then en [(x, kind)]
              (projected_closures x slot (other_slots slot decls) body))
           body)
    else rebuild (bind_variables en [(x, kind)]) body.
Proof.
  intros Hep. destruct en as [ec ek ep]. simpl in Hep. subst ep.
  reflexivity.
Qed.

Lemma rebuild_let_eq (en : rebuild_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  env_projections en = [] -> named_plain d = true ->
  rebuild en (Let x kind d body)
  = if is_dead_var sol x kind then Invalid
    else if is_var_used sol x kind
    then
      Let x kind (rebuild_named sol en x d)
        (rebuild (bind_variables en [(x, kind)]) body)
    else rebuild (bind_variables en [(x, kind)]) body.
Proof.
  intros Hep Hd. destruct en as [ec ek ep]. simpl in Hep. subst ep.
  destruct d; try discriminate Hd; reflexivity.
Qed.

Definition rec_env (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) : rebuild_env :=
  bind_continuations en
    (map (fun '(k, h) => (k, handler_mask sol false h)) handlers).

Definition rebuilt_handlers (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, rebuild_h en false (handler_mask sol false h) h))
    handlers.

Lemma rebuild_let_cont_recursive_eq (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  rebuild en (Let_cont_recursive handlers body)
  = Let_cont_recursive (rebuilt_handlers (rec_env en handlers) handlers)
      (rebuild (rec_env en handlers) body).
Proof. reflexivity. Qed.

Lemma map_fst_rebuilt_handlers (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) :
  map fst (rebuilt_handlers en handlers) = map fst handlers.
Proof.
  unfold rebuilt_handlers. rewrite map_map.
  apply map_ext. intros [k h]. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Terms of the fragment                                                    *)
(* ------------------------------------------------------------------------ *)

(* The rebuild names no continuation that the term does not. *)
Lemma free_rebuild :
  (∀ e : expr,
     checked exact codes e = true ->
     ∀ (en : rebuild_env) (k : continuation),
       k ∈ free_continuations_of_expr (rebuild en e) ->
       k ∈ free_continuations_of_expr e)
  ∧ (∀ h : continuation_handler,
       handler_checked exact codes h = true ->
       ∀ (en : rebuild_env) (dead_check : bool) (mask : list bool)
         (k : continuation),
         k ∈ free_continuations_of_handler (rebuild_h en dead_check mask h) ->
         k ∈ free_continuations_of_handler h).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Hfrag en k. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [_ Hbody].
    destruct (rebuild_let_cases en x kind d body)
      as [-> | [(d' & en' & ->) | (en' & ->)]]; [ intros [] | | ];
      simpl; apply IH; exact Hbody.
  - intros bindings body Hcodes IH Hfrag en k. simpl in Hfrag |- *.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    intros Hin. apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b & Hb & Hk).
      apply in_flat_map in Hb as (b0 & Hb0 & Hb).
      rewrite forallb_forall in Hbindings. specialize (Hbindings b0 Hb0).
      destruct b0 as [id c | id | s const].
      * cbn [rebuild_binding_cc] in Hb.
        match type of Hb with
        | context [if ?b then _ else _] => destruct b
        end; destruct Hb as [<- | []]; [ | destruct Hk ].
        apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hb0 | ].
        destruct (binding_checked_cases _ _ _ Hbindings)
          as [Hnc0 | (id' & c' & Heq & He & Hnc0)]; [ discriminate Hnc0 | ].
        injection Heq as <- <-.
        assert (Hcb : checked exact codes (code_body c) = true)
          by (rewrite He; exact Hnc0).
        pose proof (Hcodes id c Hb0 Hcb) as IHc.
        destruct c as [params mc ret exn md cbody].
        cbn [rebuild_code_cc] in Hk.
        destruct (code_calling_convention_change sol cannot_change code_deps
                    unit_code_ids id md);
          cbn [code_body] in Hk; simpl in Hk |- *;
          unfold without in Hk |- *; apply filter_In in Hk as [Hk Hnot];
          apply filter_In; (split; [ | exact Hnot ]);
          (match type of Hk with
           | context [if ?b then _ else _] => destruct b
           end; [ destruct Hk | exact (IHc _ _ Hk) ]).
      * simpl in Hb. destruct Hb as [<- | []]. destruct Hk.
      * cbn [rebuild_binding_cc] in Hb.
        match type of Hb with
        | context [if ?b then _ else _] => destruct b
        end; [ | destruct Hb ].
        destruct Hb as [<- | []]. destruct Hk.
    + apply in_or_app. right. exact (IH Hbody _ _ Hin).
  - intros k0 h body IHh IH Hfrag en k. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hh Hbody]. cbn [rebuild_expr_cc].
    set (mask := handler_mask sol (handler_is_exn_handler h) h).
    set (body' := rebuild (bind_continuations en [(k0, mask)]) body).
    simpl. intros Hin. apply in_or_app.
    destruct (among k0 (free_continuations_of_expr body')) eqn:Hk0.
    + simpl in Hin. apply in_app_or in Hin as [Hin | Hin].
      * left. exact (IHh Hh _ _ _ _ Hin).
      * right. apply among_In. apply among_In in Hin.
        rewrite among_without in Hin |- *.
        apply andb_true_iff in Hin as [Hin Hnot]. rewrite Hnot, andb_true_r.
        apply among_In. exact (IH Hbody _ _ (proj1 (among_In _ _) Hin)).
    + right. apply among_In. rewrite among_without.
      apply andb_true_intro. split.
      * apply among_In. exact (IH Hbody _ _ Hin).
      * simpl. destruct (Pos.eqb_spec k k0) as [-> | _]; [ | reflexivity ].
        apply among_In in Hin. rewrite Hin in Hk0. discriminate.
  - intros handlers body IHh IH Hfrag en k. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    rewrite rebuild_let_cont_recursive_eq. simpl.
    rewrite map_fst_rebuilt_handlers. intros Hin.
    apply among_In. apply among_In in Hin.
    rewrite among_without in Hin |- *.
    apply andb_true_iff in Hin as [Hin Hnot]. rewrite Hnot, andb_true_r.
    apply among_In. apply among_In in Hin.
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app; [ left | right ].
    + apply in_flat_map in Hin as ([k' h'] & Hkh & Hk).
      unfold rebuilt_handlers in Hkh. apply in_map_iff in Hkh
        as ([k'' h''] & [= <- <-] & Hin').
      apply in_flat_map. exists (k'', h''). split; [ exact Hin' | ].
      exact (IHh k'' h'' Hin' (Hhandlers _ Hin') _ _ _ _ Hk).
    + exact (IH Hbody _ _ Hin).
  - intros ap _ en k. cbn [rebuild_expr_cc].
    destruct (rebuild_apply_cc_cases en ap)
      as [-> | [-> | (ap' & fd & -> & Ek & Ex)]]; [ | intros [] | ].
    + unfold rebuild_apply.
      destruct (_ || _); [ intros [] | ].
      unfold make_apply_wrapper. cbn [ap_continuation].
      destruct (lookup (ap_continuation ap) (env_conts en)) as [mask | ];
        [ | exact (fun H => H) ].
      destruct (forallb _ mask); [ exact (fun H => H) | ].
      cbn [free_continuations_of_expr free_continuations_of_handler
           apply_cont_continuations with_continuation ap_continuation
           ap_exn_continuation ac_continuation ac_trap_action app].
      intros [<- | Hin]; [ left; reflexivity | right ].
      unfold without in Hin. apply filter_In in Hin as [[<- | [<- | []]] Hn].
      * unfold among in Hn. simpl in Hn. rewrite Pos.eqb_refl in Hn.
        discriminate Hn.
      * left. reflexivity.
    + unfold make_apply_wrapper_cc. cbv zeta.
      destruct (forallb _ _);
        cbn [free_continuations_of_expr free_continuations_of_handler
             apply_cont_continuations with_continuation ap_continuation
             ap_exn_continuation ac_continuation ac_trap_action app];
        rewrite Ek, Ex; [ exact (fun H => H) | ].
      intros [<- | Hin]; [ left; reflexivity | right ].
      unfold without in Hin. apply filter_In in Hin as [[<- | [<- | []]] Hn].
      * unfold among in Hn. simpl in Hn. rewrite Pos.eqb_refl in Hn.
        discriminate Hn.
      * left. reflexivity.
  - intros ac _ en k. cbn [rebuild_expr_cc]. unfold rebuild_apply_cont.
    destruct (rewrite_apply_cont sol en ac) as [ac' | ] eqn:Hrw;
      [ | intros [] ].
    unfold rewrite_apply_cont in Hrw.
    destruct (existsb (simple_has_no_source sol) (ac_args ac));
      [ discriminate Hrw | ].
    injection Hrw as <-. simpl. exact (fun H => H).
  - intros scrutinee arms _ en k. cbn [rebuild_expr_cc]. unfold rebuild_switch.
    cbv zeta.
    destruct (Keyed.is_empty _); [ intros [] | ].
    simpl. intros Hin. apply in_flat_map in Hin as ([d ac'] & Hin & Hk).
    apply in_flat_map in Hin as ([d' ac] & Hcomp & Hin).
    destruct (rewrite_apply_cont sol en ac) as [ac'' | ] eqn:Hrw;
      [ | destruct Hin ].
    destruct Hin as [[= <- <-] | []].
    apply in_flat_map. exists (d', ac). split.
    + apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hcomp.
      exact (keyed_lookup_in Z.eqb Z.eqb_eq d' ac arms Hcomp).
    + unfold rewrite_apply_cont in Hrw.
      destruct (existsb (simple_has_no_source sol) (ac_args ac));
        [ discriminate Hrw | ].
      injection Hrw as <-. exact Hk.
  - intros _ en k [].
  - intros params metadata handler IH Hfrag en dead_check mask k.
    simpl. destruct (dead_check && _); [ intros [] | ].
    apply IH. exact Hfrag.
Qed.

(* The rebuild binds nothing static that the term does not. *)
Lemma binds_statics_rebuild :
  (∀ e : expr,
     checked exact codes e = true ->
     ∀ en : rebuild_env,
       binds_statics (rebuild en e) = true -> binds_statics e = true)
  ∧ (∀ h : continuation_handler,
       handler_checked exact codes h = true ->
       ∀ (en : rebuild_env) (dead_check : bool) (mask : list bool),
         handler_binds_statics (rebuild_h en dead_check mask h) = true ->
         handler_binds_statics h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Hfrag en. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [_ Hbody].
    destruct (rebuild_let_cases en x kind d body)
      as [-> | [(d' & en' & ->) | (en' & ->)]]; [ discriminate | | ];
      simpl; apply IH; exact Hbody.
  - intros bindings body _ IH Hfrag en. simpl in Hfrag |- *.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    intros Hb. apply orb_true_iff in Hb as [Hb | Hb].
    + apply orb_true_intro. left.
      apply existsb_exists in Hb as (b & Hin & Hbinds).
      apply in_flat_map in Hin as (b0 & Hb0 & Hin).
      apply existsb_exists. exists b0. split; [ exact Hb0 | ].
      rewrite forallb_forall in Hbindings. specialize (Hbindings b0 Hb0).
      destruct b0 as [id c | id | s const]; [ reflexivity | | reflexivity ].
      simpl in Hin. destruct Hin as [<- | []]. discriminate Hbinds.
    + apply orb_true_intro. right. exact (IH Hbody _ Hb).
  - intros k0 h body IHh IH Hfrag en. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hh Hbody]. cbn [rebuild_expr_cc].
    destruct (among k0 _); simpl; intros Hb;
      [ apply orb_true_iff in Hb as [Hb | Hb]; apply orb_true_intro;
        [ left; exact (IHh Hh _ _ _ Hb) | right; exact (IH Hbody _ Hb) ]
      | apply orb_true_intro; right; exact (IH Hbody _ Hb) ].
  - intros handlers body IHh IH Hfrag en. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    rewrite rebuild_let_cont_recursive_eq. simpl. intros Hb.
    apply orb_true_iff in Hb as [Hb | Hb]; apply orb_true_intro;
      [ left | right; exact (IH Hbody _ Hb) ].
    apply existsb_exists in Hb as ([k h'] & Hin & Hbinds).
    unfold rebuilt_handlers in Hin.
    apply in_map_iff in Hin as ([k' h''] & [= <- <-] & Hin).
    apply existsb_exists. exists (k', h''). split; [ exact Hin | ].
    exact (IHh k' h'' Hin (Hhandlers _ Hin) _ _ _ Hbinds).
  - intros ap _ en. cbn [rebuild_expr_cc].
    destruct (rebuild_apply_cc_cases en ap)
      as [-> | [-> | (ap' & fd & -> & _ & _)]]; [ | discriminate | ].
    + unfold rebuild_apply.
      destruct (_ || _); [ discriminate | ].
      unfold make_apply_wrapper.
      destruct (lookup _ (env_conts en)); [ | discriminate ].
      destruct (forallb _ _); discriminate.
    + unfold make_apply_wrapper_cc. cbv zeta.
      destruct (forallb _ _); discriminate.
  - intros ac _ en. cbn [rebuild_expr_cc]. unfold rebuild_apply_cont.
    destruct (rewrite_apply_cont sol en ac); discriminate.
  - intros scrutinee arms _ en. cbn [rebuild_expr_cc]. unfold rebuild_switch.
    cbv zeta. destruct (Keyed.is_empty _); discriminate.
  - intros _ en. discriminate.
  - intros params metadata handler IH Hfrag en dead_check mask.
    simpl. destruct (dead_check && _); [ discriminate | ].
    apply IH. exact Hfrag.
Qed.

Lemma rebuild_let_static_eq (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  rebuild en (Let_static bindings body)
  = Let_static (flat_map (rebuild_b en) bindings)
      (rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma rebuild_let_cont_non_recursive_eq (en : rebuild_env)
    (k : continuation) (h : continuation_handler) (body : expr) :
  rebuild en (Let_cont_non_recursive k h body)
  = if among k
         (free_continuations_of_expr
            (rebuild
               (bind_continuations en
                  [(k, handler_mask sol (handler_is_exn_handler h) h)])
               body))
    then
      Let_cont_non_recursive k
        (rebuild_h en true (handler_mask sol (handler_is_exn_handler h) h) h)
        (rebuild
           (bind_continuations en
              [(k, handler_mask sol (handler_is_exn_handler h) h)])
           body)
    else
      rebuild
        (bind_continuations en
           [(k, handler_mask sol (handler_is_exn_handler h) h)])
        body.
Proof. reflexivity. Qed.

Lemma rebuild_handler_eq (en : rebuild_env) (dead_check : bool)
    (mask : list bool) (params : list variable) (metadata : handler_metadata)
    (handler : expr) :
  rebuild_h en dead_check mask (Handler params metadata handler)
  = Handler (filter_kept mask params)
      (with_params_kinds metadata
         (filter_kept mask (hm_params_kinds metadata)))
      (if dead_check
          && existsb (fun '(x, kind) => is_dead_var sol x kind)
               (with_kinds params (hm_params_kinds metadata))
       then Invalid
       else
         rebuild
           (bind_variables en (with_kinds params (hm_params_kinds metadata)))
           handler).
Proof. reflexivity. Qed.

(* The [Apply_cont] that the rebuild makes of one whose arguments have
   sources. *)
Definition rebuilt_apply_cont (en : rebuild_env) (ac : apply_cont)
    : apply_cont :=
  {| ac_continuation := ac_continuation ac;
     ac_args :=
       match lookup (ac_continuation ac) (env_conts en) with
       | Some mask =>
         filter_kept mask (map (rewrite_simple sol en) (ac_args ac))
       | None => map (rewrite_simple sol en) (ac_args ac)
       end;
     ac_trap_action := ac_trap_action ac |}.

Lemma rewrite_apply_cont_some (en : rebuild_env) (ac : apply_cont) :
  existsb (simple_has_no_source sol) (ac_args ac) = false ->
  rewrite_apply_cont sol en ac = Some (rebuilt_apply_cont en ac).
Proof. intros H. unfold rewrite_apply_cont. rewrite H. reflexivity. Qed.

Lemma eval_simples_member (vars : env) (args : list simple)
    (vs : list value) (s : simple) :
  eval_simples vars args = Some vs -> s ∈ args ->
  ∃ w : value, eval_simple vars s = Some w.
Proof.
  revert vs. induction args as [ | s' args IH]; intros vs Hvs Hin;
    [ destruct Hin | ]. simpl in Hvs.
  destruct (eval_simple vars s') as [w | ] eqn:Hs; [ | discriminate ].
  destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
  destruct Hin as [<- | Hin];
    [ exists w; exact Hs | exact (IH ws eq_refl Hin) ].
Qed.

(* The arguments of an [Apply_cont] that the first run evaluates have
   sources. *)
Lemma args_have_sources (vars : env) (args : list simple) (vs : list value) :
  env_flows_ vars -> simples_noted noted args ->
  eval_simples vars args = Some vs ->
  existsb (simple_has_no_source sol) args = false.
Proof.
  intros Hvars Hnoted Hvs. apply not_true_iff_false. intros Hex.
  apply existsb_exists in Hex as (s & Hin & Hno).
  destruct (eval_simples_member vars args vs s Hvs Hin) as (w & Hw).
  pose proof (flows_hs is_local G sol keep called no_loc Hc _ _
                (simple_flows is_local G sol noted cds keep no_loc Hc Hconsts
                    Hsyms _ _
                   _ Hvars (Hnoted s Hin) Hw)) as Hhs.
  destruct s; simpl in Hno, Hhs; try discriminate Hno;
    rewrite Hhs in Hno; discriminate Hno.
Qed.

Lemma bind_params_length (params : list variable) :
  ∀ (vs : list value) (vars hv : env),
    bind_params params vs vars = Some hv -> length vs = length params.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars hv H;
    simpl in H; try discriminate; [ reflexivity | ].
  simpl. f_equal. exact (IH vs _ _ H).
Qed.

Lemma with_kinds_in (params : list variable) :
  ∀ (kinds : list flambda_kind) (x : variable) (k : flambda_kind),
    (x, k) ∈ with_kinds params kinds -> x ∈ params.
Proof.
  induction params as [ | p params IH]; intros [ | k0 kinds] x k Hin;
    simpl in Hin; try contradiction;
    (destruct Hin as [[= -> _] | Hin]; [ left; reflexivity | right ];
     exact (IH _ x k Hin)).
Qed.

(* The parameters of a handler entered have sources: its rebuild is not
   [Invalid]. *)
Lemma params_not_dead (params : list variable) (kinds : list flambda_kind)
    (vs : list value) (vars hv : env) :
  bind_params params vs vars = Some hv -> env_flows_ hv ->
  existsb (fun '(x, kind) => is_dead_var sol x kind)
    (with_kinds params kinds) = false.
Proof.
  intros Hbind Hhv. apply not_true_iff_false. intros Hex.
  apply existsb_exists in Hex as ([x kind] & Hin & Hdead).
  destruct (bind_params_bound params vs vars hv x Hbind
              (or_introl (with_kinds_in _ _ _ _ Hin)))
    as (w & Hw).
  pose proof (flows_hs is_local G sol keep called no_loc Hc _ _ (Hhv x w Hw))
      as Hhs.
  unfold is_dead_var in Hdead. rewrite Hhs in Hdead.
  destruct kind; discriminate Hdead.
Qed.

(* Entering a handler in the two runs: the second, at the parameters and
   values kept, under an environment related to the first's. *)
Lemma enter_handler (b : bool) (params : list variable)
    (metadata : handler_metadata) (handler body' : expr) (en0 : rebuild_env)
    (vars0 vars0' hv : env) (conts0' : cont_env)
    (rec' : list (continuation * continuation_handler))
    (vs vs' : list value) (statics' : static_env) (hp' : heap)
    (rcd' : read_record) :
  env_ok (env_kinds en0) vars0 vars0' ->
  map kind_of_value vs = hm_params_kinds metadata ->
  map kind_of_value vs' = map kind_of_value vs ->
  bind_params params vs vars0 = Some hv ->
  (∀ (i : nat) (p : variable) (v v' : value),
     nth_error params i = Some p -> nth_error vs i = Some v ->
     nth_error vs' i = Some v' ->
     is_var_used sol p (kind_of_value v) = true ->
     rel v (Node_variable p) v') ->
  ∃ hv' : env,
    apply_cont_value
      (Closure
         (Handler
            (filter_kept (handler_mask sol b (Handler params metadata handler))
               params)
            (with_params_kinds metadata
               (filter_kept
                  (handler_mask sol b (Handler params metadata handler))
                  (hm_params_kinds metadata)))
            body')
         vars0' conts0' rec')
      (filter_kept (handler_mask sol b (Handler params metadata handler)) vs')
      statics' hp' rcd'
    = Step (State body' hv' (rec_closures rec' vars0' conts0' ++ conts0')
              statics' hp' rcd')
    ∧ env_ok
        (env_kinds
           (bind_variables en0 (with_kinds params (hm_params_kinds metadata))))
        hv hv'.
Proof.
  intros Henv0 Hkinds Hkinds' Hbind Hrel.
  set (mask := handler_mask sol b (Handler params metadata handler)).
  pose proof (bind_params_length params vs vars0 hv Hbind) as Hlen.
  assert (Hlen' : length vs' = length vs).
  { rewrite <- (length_map kind_of_value vs'), Hkinds', length_map.
    reflexivity. }
  destruct (bind_params_rel params vs vs' mask (env_kinds en0) vars0 vars0' hv
              Henv0 Hlen' (handler_mask_length b params metadata handler))
    as (hv' & Hbind' & Hok); [ | exact Hbind | ].
  { intros i p v v' Hp Hv Hv' Hu.
    split; [ | exact (Hrel i p v v' Hp Hv Hv' Hu) ].
    apply (handler_mask_used b params metadata handler i p (kind_of_value v)
             Hp);
      [ | exact Hu ].
    rewrite <- Hkinds, nth_error_map, Hv. reflexivity. }
  exists hv'. split.
  - cbn [apply_cont_value]. rewrite with_params_kinds_kinds.
    rewrite <- filter_kept_map, Hkinds', Hkinds, kinds_eqb_refl.
    rewrite Hbind'. reflexivity.
  - cbn [env_kinds bind_variables bind_variables_then].
    rewrite (with_kinds_combine params (hm_params_kinds metadata)),
      <- Hkinds; [ exact Hok | ].
    rewrite <- Hkinds, length_map. exact Hlen.
Qed.

(* ------------------------------------------------------------------------ *)
(* The relation                                                             *)
(* ------------------------------------------------------------------------ *)

Definition unit_cont (k : continuation) : Prop :=
  k = unit_return u ∨ k = unit_exn u.

(* The nodes of the parameters of a handler. *)
Definition cont_params (c : cont_value) : list node :=
  match c with
  | Closure h _ _ _ => handler_params h
  | _ => []
  end.

(* The nodes [ns] of a way out of the unit through its continuation
   [k0]: each has a use, and what is related at it is related at the
   parameter of [k0]. *)
Definition outer_nodes (k0 : continuation) (ns : list node) : Prop :=
  length ns = length (unit_cont_nodes u k0)
  ∧ ∀ (i : nat) (n p : node),
      nth_error ns i = Some n -> nth_error (unit_cont_nodes u k0) i = Some p ->
      hu n ∧ ∀ v v' : value, rel v n v' -> rel v p v'.

(* The kinds of the parameters of a handler. *)
Definition cont_kinds (c : cont_value) : list flambda_kind :=
  match c with
  | Closure h _ _ _ => handler_kinds h
  | _ => []
  end.

(* The nodes [ns] of the results of code that returns to the handler [c]:
   what is related at one of them is related at the parameter of [c]. *)
Definition returns_to (ns : list node) (c : cont_value) : Prop :=
  ∀ (i : nat) (n p : node),
    nth_error ns i = Some n -> nth_error (cont_params c) i = Some p ->
    (hu p -> hu n) ∧ ∀ v v' : value, rel v n v' -> rel v p v'.

(* What is related at [n'] is related at [n], and [n'] has a use where [n]
   has one. *)
Definition transports (n' n : node) : Prop :=
  (hu n -> hu n') ∧ ∀ v v' : value, rel v n' v' -> rel v n v'.

(* The allocations that may come to [n], each an allocation of its
   own. *)
Definition from_node (n y : node) : Prop :=
  may_come_from sol y n ∧ may_come_from sol y y.

(* The code of the unit, as in a unit with a call of unknown arity
   (FlambdaReaperFlowProofs.indirect_ok): if not tupled, it has
   parameters, and if tupled, parameters of one component each. *)
Definition indirect_codes : Prop := indirect_ok codes.

(* The arguments [vs] of a call at the arity [a], and [vs'] in the second
   run: of the same kinds, each related at the parameters that have a
   use of the witnesses that its group reaches from the allocations of
   [P]. *)
Definition args_rel (P : node -> Prop) (a : arity) (vs vs' : list value)
    : Prop :=
  map kind_of_value vs' = map kind_of_value vs
  ∧ fields_related
      (fun i v v' =>
         ∀ (g k : nat) (W p : node),
           group_position (arity_sizes a) i = Some (g, k) ->
           witness_chain G sol P g W -> g_parameter G W k p -> hu p ->
           rel v p v')
      0 vs vs'.

(* The witnesses [ws] of a call of unknown arity, which reads its
   results at [returns] and its exception at [exn]: each stands for the
   witnesses that its group reaches from the allocations of [P]; the
   last gives the results, each the exception, each but the last a first
   result whose call witness is the next; and each reads the code id of
   what it stands for into a node with any usage. *)
Definition call_reads (P : node -> Prop) (ws returns : list node)
    (exn : node) : Prop :=
  (∀ (g : nat) (W w : node),
     witness_chain G sol P g W -> nth_error ws g = Some w ->
     may_come_from sol W w)
  ∧ (∀ (i : nat) (w r : node),
       nth_error ws (pred (length ws)) = Some w ->
       nth_error returns i = Some r ->
       g_accessor G r (Field_return_of_call (Normal i)) w)
  ∧ (∀ (g : nat) (w : node),
       nth_error ws g = Some w ->
       g_accessor G exn (Field_return_of_call Exn) w)
  ∧ (∀ (g : nat) (w w' : node),
       nth_error ws g = Some w -> nth_error ws (S g) = Some w' ->
       ∃ v : node,
         g_accessor G v (Field_return_of_call (Normal 0)) w
         ∧ g_accessor G w' (Field_call_witness Unknown_arity_code_pointer) v)
  ∧ (∃ called_node : node,
       au called_node
       ∧ ∀ (g : nat) (w : node),
           nth_error ws g = Some w ->
           g_accessor G called_node Field_code_id_of_call_witness w)
  ∧ ∀ (W w : node),
      witness_chain G sol P 0 W -> nth_error ws 0 = Some w -> hu W.

(* The wrapper of [make_apply_wrapper], defined under [vars] and
   [conts]. *)
Definition wrapper (k : continuation) (mask : list bool)
    (kinds : list flambda_kind) (vars : env) (conts : cont_env)
    : cont_value :=
  Closure
    (Handler (names_from (length kinds) first_fresh)
       (make_handler_metadata kinds)
       (Apply_cont
          {| ac_continuation := k;
             ac_args :=
               filter_kept mask
                 (map Var (names_from (length kinds) first_fresh));
             ac_trap_action := None |}))
    vars conts [].

(* The extra arguments of an exception in the two runs, at the parameters
   of the handler [c] after the first: of the same kinds, and related
   where the rebuild keeps the parameter. *)
Definition extras_related (c : cont_value) (extras extras' : list value)
    : Prop :=
  map kind_of_value extras' = map kind_of_value extras
  ∧ ∀ (i : nat) (p : variable) (v v' : value),
      nth_error (cont_params c) (S i) = Some (Node_variable p) ->
      nth_error extras i = Some v -> nth_error extras' i = Some v' ->
      is_var_used sol p (kind_of_value v) = true ->
      rel v (Node_variable p) v'.

(* A continuation that no handler of the expression binds, at the nodes
   [ns], in the two runs: a way out of the unit through one of its
   continuations; a handler that code returns to, related by [R] to what
   the second run has, or to what the wrapper of the call applies
   ([make_apply_wrapper]); a handler that code raises to, at the node
   [n] of the exception, related by [R] to what the second run has,
   keeping the exception, with the extra arguments of the call, those
   kept in the second run; or what is left of an over-application,
   related by [R2] at a node to which the first of [ns] transports. *)
Definition way_out (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (ns : list node) (c c' : cont_value) : Prop :=
  (∃ k0 : continuation,
     c = Outer k0 ∧ c' = Outer k0 ∧ unit_cont k0 ∧ outer_nodes k0 ns)
  ∨ (∃ (mask : list bool) (c'' : cont_value),
       returns_to ns c ∧ R mask c c''
       ∧ ((c' = c'' ∧ forallb (fun keep => keep) mask = true)
          ∨ ∃ (k : continuation) (vars : env) (conts : cont_env),
              c' = wrapper k mask (cont_kinds c) vars conts
              ∧ lookup k conts = Some c''))
  ∨ (∃ (n : node) (mask : list bool) (h h' : cont_value)
       (extras extras' : list value),
       ns = [n] ∧ c = with_extra_args h extras
       ∧ c' = with_extra_args h' (filter_kept (tl mask) extras')
       ∧ returns_to [n] h ∧ R mask h h' ∧ hd true mask = true
       ∧ extras_related h extras extras')
  ∨ ∃ (n n0 : node) (ns0 : list node),
      ns = n :: ns0 ∧ transports n n0 ∧ R2 n0 c c'.

(* The wrapper of [make_apply_wrapper_cc], defined under [vars] and
   [conts]: it takes the results that the callee keeps, and gives the
   continuation [k] of the call those it keeps ([amask]), poison where the
   callee no longer returns one. *)
Definition cc_wrapper (k : continuation) (amask : list bool)
    (fd : list (param_decision * flambda_kind)) (vars : env)
    (conts : cont_env) : cont_value :=
  Closure
    (Handler (names_from (length (kept_kinds (map fst fd))) first_fresh)
       (make_handler_metadata (kept_kinds (map fst fd)))
       (Apply_cont
          {| ac_continuation := k;
             ac_args := wrapper_args first_fresh amask fd;
             ac_trap_action := None |}))
    vars conts [].

(* The return continuation of code that changes its calling convention,
   at the nodes [ns] of its results, of the kinds [K], with the mask [m]
   of the results that it keeps, in the two runs: a way out, where every
   result is kept; the handler that the call returns to, related by [R]
   at [m] to what the second run has; the wrapper of the call
   ([make_apply_wrapper_cc]) over the handler, related by [R] at the
   mask [hm] of the handler: the wrapper gives the handler what it keeps
   of what the code returns, and poison where the code no longer returns
   what the handler keeps, which then has no use there; or the wrapper
   of a call in tail position over a way out at the nodes [nsw], to which
   it gives every result, poison where the code no longer returns one,
   which then has no use at [nsw]. *)
Definition ret_base (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (m : list bool) (K : list flambda_kind) (ns : list node)
    (c c' : cont_value) : Prop :=
  (forallb (fun keep => keep) m = true ∧ way_out R R2 ns c c')
  ∨ (∃ hm : list bool,
       returns_to ns c ∧ R hm c c' ∧ m = pad_mask (length ns) hm)
  ∨ (∃ (k : continuation) (hm : list bool) (rd : list param_decision)
       (kinds : list flambda_kind) (vars : env) (conts : cont_env)
       (c'' : cont_value),
       c' = cc_wrapper k (pad_mask (length rd) hm) (combine rd kinds) vars
              conts
       ∧ lookup k conts = Some c''
       ∧ returns_to ns c ∧ R hm c c''
       ∧ m = map is_kept rd
       ∧ kinds = K
       ∧ length rd = length kinds
       ∧ (∀ (i : nat) (kind : flambda_kind),
            nth_error rd i = Some (Keep kind) ->
            nth_error kinds i = Some kind)
       ∧ (∀ (i : nat) (p : variable) (kind : flambda_kind),
            nth_error rd i = Some Delete ->
            nth_error (cont_params c) i = Some (Node_variable p) ->
            nth_error kinds i = Some kind ->
            is_var_used sol p kind = false))
  ∨ ∃ (k : continuation) (rd : list param_decision) (vars : env)
      (conts : cont_env) (nsw : list node),
      c' = cc_wrapper k (pad_mask (length rd) []) (combine rd K) vars conts
      ∧ way_out R R2 nsw c (resolve k conts)
      ∧ m = map is_kept rd
      ∧ length rd = length K
      ∧ length nsw = length ns
      ∧ length ns = length rd
      ∧ (∃ i : nat, nth_error rd i = Some Delete)
      ∧ (∀ (i : nat) (n n' : node),
           nth_error ns i = Some n -> nth_error nsw i = Some n' ->
           transports n n')
      ∧ (∀ (i : nat) (kind : flambda_kind),
           nth_error rd i = Some (Keep kind) -> nth_error K i = Some kind)
      ∧ ∀ i : nat,
          nth_error rd i = Some Delete ->
          (∀ n' : node, nth_error nsw i = Some n' -> ~ hu n')
          ∧ ∀ kind : flambda_kind,
              nth_error K i = Some kind -> kind <> Region ∧ kind <> Rec_info.

(* [ret_base], or the wrapper of a call in tail position in code that
   changes over the return continuation of the code, which keeps other
   results ([m0]): it gives the return continuation the results it keeps,
   poison where the code called no longer returns one, which then has no
   use at the nodes [ns0] of the return continuation. *)
Inductive ret_rel (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    : list bool -> list flambda_kind -> list node -> cont_value
      -> cont_value -> Prop :=
  | Ret_base (m : list bool) (K : list flambda_kind) (ns : list node)
      (c c' : cont_value) :
    ret_base R R2 m K ns c c' -> ret_rel R R2 m K ns c c'
  | Ret_wrapper_ret (m0 : list bool) (K : list flambda_kind)
      (ns ns0 : list node) (c : cont_value) (k : continuation)
      (rd : list param_decision) (vars : env) (conts : cont_env) :
    ret_rel R R2 m0 K ns0 c (resolve k conts) ->
    length rd = length K -> length ns0 = length ns -> length ns = length rd ->
    (∃ i : nat, nth_error m0 i = Some false ∨ nth_error rd i = Some Delete) ->
    (∀ (i : nat) (n n' : node),
       nth_error ns i = Some n -> nth_error ns0 i = Some n' ->
       transports n n') ->
    (∀ (i : nat) (kind : flambda_kind),
       nth_error rd i = Some (Keep kind) -> nth_error K i = Some kind) ->
    (∀ i : nat,
       nth_error rd i = Some Delete ->
       (∀ n' : node, nth_error ns0 i = Some n' -> ~ hu n')
       ∧ ∀ kind : flambda_kind,
           nth_error K i = Some kind -> kind <> Region ∧ kind <> Rec_info) ->
    ret_rel R R2 (map is_kept rd) K ns c
      (cc_wrapper k (pad_mask (length rd) m0) (combine rd K) vars conts).

Lemma ret_wrapper_ret_intro
    (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (m0 : list bool) (K : list flambda_kind) (ns ns0 : list node)
    (c c' : cont_value) (k : continuation) (rd : list param_decision)
    (vars : env) (conts : cont_env) :
  c' = cc_wrapper k (pad_mask (length rd) m0) (combine rd K) vars conts ->
  ret_rel R R2 m0 K ns0 c (resolve k conts) ->
  length rd = length K -> length ns0 = length ns -> length ns = length rd ->
  (∃ i : nat, nth_error m0 i = Some false ∨ nth_error rd i = Some Delete) ->
  (∀ (i : nat) (n n' : node),
     nth_error ns i = Some n -> nth_error ns0 i = Some n' ->
     transports n n') ->
  (∀ (i : nat) (kind : flambda_kind),
     nth_error rd i = Some (Keep kind) -> nth_error K i = Some kind) ->
  (∀ i : nat,
     nth_error rd i = Some Delete ->
     (∀ n' : node, nth_error ns0 i = Some n' -> ~ hu n')
     ∧ ∀ kind : flambda_kind,
         nth_error K i = Some kind -> kind <> Region ∧ kind <> Rec_info) ->
  ret_rel R R2 (map is_kept rd) K ns c c'.
Proof.
  intros -> Hin. exact (Ret_wrapper_ret R R2 m0 K ns ns0 c k rd vars conts Hin).
Qed.

(* The return continuations [rets] of code that changes, with the kinds
   of its results, but those of [ks]. *)
Definition drop_rets (ks : list continuation)
    (rets : finite_map (list flambda_kind)) : finite_map (list flambda_kind) :=
  filter (fun '(r, _) => negb (among r ks)) rets.

(* The continuations [ks] in the two runs, with the masks of the rebuild
   and the nodes of the parameters ([ce]): ways out of the term or of the
   code, or continuations related by [R]. *)
Definition conts_related (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (rets : finite_map (list flambda_kind))
    (ks : list continuation) (conts conts' : cont_env)
    (masks : finite_map (list bool)) (ce : finite_map (list node)) : Prop :=
  ∀ k : continuation,
    k ∈ ks ->
    (lookup k masks = None
     ∧ ∃ ns : list node,
         lookup k ce = Some ns
         ∧ way_out R R2 ns (resolve k conts) (resolve k conts'))
    ∨ (∃ (mask : list bool) (c c' : cont_value),
         lookup k masks = Some mask ∧ lookup k conts = Some c
         ∧ lookup k conts' = Some c' ∧ lookup k ce = Some (cont_params c)
         ∧ R mask c c' ∧ lookup k rets = None)
    ∨ (∃ K : list flambda_kind,
         lookup k rets = Some K
         ∧ ∃ (m : list bool) (ns : list node),
             (lookup k masks = Some m
              ∨ (lookup k masks = None
                 ∧ forallb (fun keep => keep) m = true))
             ∧ length m = length K
             ∧ lookup k ce = Some ns
             ∧ ret_rel R R2 m K ns (resolve k conts) (resolve k conts')).

(* What the unit asks of an expression. *)
Definition live_ (ctx : option code_id) : Prop :=
  match ctx with
  | None => True
  | Some id => au (Node_code_id id)
  end.

(* The handlers in scope ([masks]) but the return continuations [rets]
   are those of the first run, of the kinds that [ks] has. *)
Definition kinds_agree (rets : finite_map (list flambda_kind))
    (masks : finite_map (list bool))
    (ks : finite_map (list flambda_kind)) (conts : cont_env) : Prop :=
  ∀ (k : continuation) (mask : list bool),
    lookup k masks = Some mask -> lookup k rets = None ->
    ∃ c : cont_value,
      lookup k conts = Some c ∧ lookup k ks = Some (cont_kinds c).

(* The handlers in scope ([masks]) are those of [xs], and an exception
   handler keeps its first parameter. *)
Definition exn_agree (masks : finite_map (list bool)) (xs : finite_map bool)
    : Prop :=
  ∀ k : continuation,
    match lookup k masks, lookup k xs with
    | Some mask, Some b => b = true -> hd true mask = true
    | None, None => True
    | _, _ => False
    end.

(* The expression names no continuation [first_fresh], its calls that
   return to handlers in scope return values of their kinds, and its
   calls that raise to handlers in scope raise to exception handlers. *)
Definition fresh_kinded (rets : finite_map (list flambda_kind))
    (masks : finite_map (list bool)) (conts : cont_env)
    (e : expr) : Prop :=
  among first_fresh (continuations_of e) = false
  ∧ (∃ ks : finite_map (list flambda_kind),
       calls_kinded ks e = true ∧ kinds_agree rets masks ks conts)
  ∧ ∃ xs : finite_map bool, raises_kept xs e = true ∧ exn_agree masks xs.

Definition handler_fresh_kinded (rets : finite_map (list flambda_kind))
    (masks : finite_map (list bool))
    (conts : cont_env) (h : continuation_handler) : Prop :=
  among first_fresh (continuations_in_handler h) = false
  ∧ (∃ ks : finite_map (list flambda_kind),
       handler_calls_kinded ks h = true ∧ kinds_agree rets masks ks conts)
  ∧ ∃ xs : finite_map bool,
      handler_raises_kept xs h = true ∧ exn_agree masks xs.

(* The continuations of the rebuild that the expression counts the
   values of: those of [rets] are. *)
Definition bound_ok (rets : finite_map (list flambda_kind))
    (en : rebuild_env) (bound : list continuation) : Prop :=
  (∀ k : continuation,
     among k bound = true -> among k (map fst (env_conts en)) = true)
  ∧ ∀ (r : continuation) (K : list flambda_kind),
      lookup r rets = Some K -> among r bound = false.

(* The return continuations of code that changes its calling convention
   ([rets]) are given as many values as counted, no call raises to them,
   and what returns there is as [ret_calls_ok] says. *)
Definition expr_ok (ctx : option code_id)
    (rets : finite_map (list flambda_kind))
    (en : rebuild_env) (ce : finite_map (list node)) (conts : cont_env)
    (e : expr) : Prop :=
  live_ ctx
  ∧ (∃ bound : list continuation,
       bound_ok rets en bound ∧ exits_bound bound ce e = true)
  ∧ scrutinees_used G ctx e ∧ checked exact codes e = true
  ∧ fresh_kinded rets (env_conts en) conts e
  ∧ ∀ (r : continuation) (K : list flambda_kind),
      lookup r rets = Some K ->
      no_raise_to r e = true ∧ rcalls_ok (env_kinds en) r K e = true.

Definition handler_ok (ctx : option code_id)
    (rets : finite_map (list flambda_kind))
    (en : rebuild_env) (ce : finite_map (list node)) (conts : cont_env)
    (h : continuation_handler) : Prop :=
  live_ ctx
  ∧ (∃ bound : list continuation,
       bound_ok rets en bound ∧ handler_exits_bound bound ce h = true)
  ∧ handler_scrutinees_used G ctx h ∧ handler_checked exact codes h = true
  ∧ handler_fresh_kinded rets (env_conts en) conts h
  ∧ ∀ (r : continuation) (K : list flambda_kind),
      lookup r rets = Some K ->
      handler_no_raise_to r h = true
      ∧ handler_ret_calls_ok (env_kinds en) r K h
        = true.

(* What a call that returns at [returns], at the arity [ret], returns
   to in the two runs: a way out at the nodes of the code that the call
   runs, where these give what is related to [returns]; and, if the call
   returns one value, a node for it, unless [c] takes no value. *)
Definition ret_ok (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (returns : list node) (ret : arity) (c c' : cont_value) : Prop :=
  ((∀ ns : list node,
      length ns = length (unarize ret) ->
      (∀ (i : nat) (n n' : node),
         nth_error returns i = Some n -> nth_error ns i = Some n' ->
         transports n' n) ->
      way_out R R2 ns c c')
   ∨ ∀ ns : list node,
       length ns = length (unarize ret) ->
       (∀ (i : nat) (n n' : node),
          nth_error returns i = Some n -> nth_error ns i = Some n' ->
          transports n' n) ->
       ret_rel R R2 (repeat true (length (unarize ret))) (unarize ret) ns c
         c')
  ∧ (length (unarize ret) = 1 ->
     returns <> []
     ∨ ∀ (v : value) (statics : static_env) (hp : heap) (rcd : read_record),
         apply_cont_value c [v] statics hp rcd = Undefined).

(* A handler of the first run, and its rebuild in the second, with the
   mask of the parameters kept; the facts of the handler are in the
   graph, under the nodes of the continuations of its definition. What
   is left of an over-application, at the node [n0] of the value it is
   given, and the same in the second run, with the arguments left
   related at the witnesses that their groups reach from what [n0] may
   come from, which the witnesses of the call stand for, and the
   continuations of the call related ([over_rel]). *)
#[local] Set Warnings "-register-all".
Inductive cont_rel : list bool -> cont_value -> cont_value -> Prop :=
  | Cont_rel_non_recursive (h : continuation_handler) (en : rebuild_env)
      (vars vars' : env) (conts conts' : cont_env) (ctx : option code_id)
      (ce : finite_map (list node)) (rets : finite_map (list flambda_kind)) :
    env_ok (env_kinds en) vars vars' ->
    handler_ok ctx rets en ce conts h ->
    handler_edges_ ctx ce h ->
    conts_related cont_rel over_rel rets
      (free_continuations_of_handler
         (rebuild_h en true (handler_mask sol (handler_is_exn_handler h) h)
            h))
      conts conts' (env_conts en) ce ->
    cont_rel (handler_mask sol (handler_is_exn_handler h) h)
      (Closure h vars conts [])
      (Closure
         (rebuild_h en true (handler_mask sol (handler_is_exn_handler h) h)
            h)
         vars' conts' [])
  | Cont_rel_recursive (k : continuation) (h : continuation_handler)
      (en : rebuild_env) (vars vars' : env) (conts conts' : cont_env)
      (handlers : list (continuation * continuation_handler))
      (ctx : option code_id) (ce : finite_map (list node))
      (rets : finite_map (list flambda_kind)) :
    env_ok (env_kinds en) vars vars' ->
    (k, h) ∈ handlers ->
    (∀ (k' : continuation) (h' : continuation_handler),
       (k', h') ∈ handlers ->
       handler_ok ctx (drop_rets (map fst handlers) rets)
         (rec_env en handlers) (rec_conts ce handlers)
         (rec_closures handlers vars conts ++ conts) h'
       ∧ handler_edges_ ctx (rec_conts ce handlers) h'
       ∧ conts_related cont_rel over_rel rets
           (without (map fst handlers)
              (free_continuations_of_handler
                 (rebuild_h (rec_env en handlers) false
                    (handler_mask sol false h') h')))
           conts conts' (env_conts en) ce) ->
    cont_rel (handler_mask sol false h)
      (Closure h vars conts handlers)
      (Closure
         (rebuild_h (rec_env en handlers) false (handler_mask sol false h) h)
         vars' conts' (rebuilt_handlers (rec_env en handlers) handlers))
with over_rel : node -> cont_value -> cont_value -> Prop :=
  | Over_rel (n0 : node) (remaining remaining' : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value)
      (ws returns : list node) (exn : node) :
    exact = false -> indirect_codes -> hu n0 ->
    remaining_arity <> [] ->
    length ws = length remaining_arity ->
    args_rel (from_node n0) remaining_arity remaining remaining' ->
    call_reads (from_node n0) ws returns exn ->
    ret_ok cont_rel over_rel returns return_arity return_to return_to' ->
    way_out cont_rel over_rel [exn] exn_to exn_to' ->
    over_rel n0
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining' remaining_arity return_arity return_to'
         exn_to').

(* The projections that the rebuild has pending are those of the closures
   of a set that begin the expression, from a variable bound to the same
   closure in the two runs. *)
Definition pending_ok (en : rebuild_env) (vars vars' : env) (e : expr)
    : Prop :=
  env_projections en = []
  ∨ ∃ (x : variable) (slot : function_slot) (others : list function_slot)
      (closure closure' : value),
      env_projections en = projected_closures x slot others e
      ∧ lookup x vars = Some closure ∧ lookup x vars' = Some closure'
      ∧ rel closure (Node_variable x) closure'
      ∧ ∃ (slot0 : function_slot) (decls : list (function_slot * code_id))
          (vs : list (value_slot * value)),
          closure = Value_closure_of_set slot0 decls vs.

(* The two runs: the first has the invariant of the flow, and the facts
   of its expression under the nodes [ce]; the expression of the second
   is the rebuild of the first's; the variables that the rebuild keeps,
   the continuations named and the static environments are related; the
   heaps are the same; and the first expression is as the unit asks. *)
Definition related (s t : state) : Prop :=
  flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc s
  ∧ ∃ (ctx : option code_id) (en : rebuild_env)
      (ce : finite_map (list node)) (rets : finite_map (list flambda_kind)),
      edges ctx ce (state_expr s)
      ∧ state_expr t = rebuild en (state_expr s)
      ∧ env_ok (env_kinds en) (state_vars s) (state_vars t)
      ∧ conts_related cont_rel over_rel rets
          (free_continuations_of_expr (state_expr t)) (state_conts s)
          (state_conts t) (env_conts en) ce
      ∧ statics_rel (state_statics s) (state_statics t)
      ∧ state_heap t = state_heap s
      ∧ expr_ok ctx rets en ce (state_conts s) (state_expr s)
      ∧ pending_ok en (state_vars s) (state_vars t) (state_expr s).

Lemma conts_related_sub (rets : finite_map (list flambda_kind))
    (ks ks' : list continuation) (conts conts' : cont_env)
    (masks : finite_map (list bool)) (ce : finite_map (list node)) :
  conts_related cont_rel over_rel rets ks conts conts' masks ce ->
  (∀ k : continuation, k ∈ ks' -> k ∈ ks) ->
  conts_related cont_rel over_rel rets ks' conts conts' masks ce.
Proof. intros H Hsub k Hk. exact (H k (Hsub k Hk)). Qed.

(* Fewer return continuations of code that changes. *)
Lemma conts_related_rets (rets rets' : finite_map (list flambda_kind))
    (ks : list continuation) (conts conts' : cont_env)
    (masks : finite_map (list bool)) (ce : finite_map (list node)) :
  conts_related cont_rel over_rel rets ks conts conts' masks ce ->
  (∀ r : continuation, r ∈ ks -> lookup r rets' = lookup r rets) ->
  conts_related cont_rel over_rel rets' ks conts conts' masks ce.
Proof.
  intros H Hsub k Hk.
  destruct (H k Hk)
    as [A | [(mask & c & c' & B1 & B2 & B3 & B4 & B5 & B6) | (K & HK & C)]];
    [ left; exact A | | ].
  - right. left. exists mask, c, c'.
    rewrite (Hsub k Hk). repeat split; assumption.
  - right. right. exists K. rewrite (Hsub k Hk). split; [ exact HK | exact C ].
Qed.

Lemma lookup_drop_rets (r : continuation) (ks : list continuation)
    (rets : finite_map (list flambda_kind)) :
  lookup r (drop_rets ks rets)
  = if among r ks then None else lookup r rets.
Proof.
  unfold drop_rets.
  induction rets as [ | [r' K'] rets IH]; cbn [filter lookup].
  - destruct (among r ks); reflexivity.
  - destruct (Pos.eqb_spec r r') as [-> | Hne].
    + destruct (among r' ks) eqn:E; cbn [negb]; [ exact IH | ].
      cbn [lookup]. rewrite Pos.eqb_refl. reflexivity.
    + destruct (among r' ks); cbn [negb lookup]; [ exact IH | ].
      rewrite (proj2 (Pos.eqb_neq _ _) Hne). exact IH.
Qed.

Lemma lookup_drop_rets_some (r : continuation) (ks : list continuation)
    (rets : finite_map (list flambda_kind)) (K : list flambda_kind) :
  lookup r (drop_rets ks rets) = Some K ->
  lookup r rets = Some K ∧ among r ks = false.
Proof.
  rewrite lookup_drop_rets. destruct (among r ks);
    [ discriminate | intros H; split; [ exact H | reflexivity ] ].
Qed.

Lemma lookup_drop_rets_none (r : continuation) (ks : list continuation)
    (rets : finite_map (list flambda_kind)) :
  lookup r rets = None ∨ among r ks = true ->
  lookup r (drop_rets ks rets) = None.
Proof.
  intros H. rewrite lookup_drop_rets.
  destruct (among r ks) eqn:E; [ reflexivity | ].
  destruct H as [H | H]; [ exact H | discriminate H ].
Qed.

Lemma conts_rel_handler_drop (mask : list bool) (c c' : cont_value)
    (k : continuation) (ks : list continuation)
    (rets : finite_map (list flambda_kind)) (conts conts' : cont_env)
    (masks : finite_map (list bool)) (ce : finite_map (list node)) :
  (lookup k masks = Some mask ∧ lookup k conts = Some c
   ∧ lookup k conts' = Some c' ∧ lookup k ce = Some (cont_params c)
   ∧ cont_rel mask c c' ∧ lookup k rets = None) ->
  lookup k masks = Some mask ∧ lookup k conts = Some c
  ∧ lookup k conts' = Some c' ∧ lookup k ce = Some (cont_params c)
  ∧ cont_rel mask c c' ∧ lookup k (drop_rets ks rets) = None.
Proof.
  intros (B1 & B2 & B3 & B4 & B5 & B6).
  repeat (split; [ assumption | ]).
  apply lookup_drop_rets_none. left. exact B6.
Qed.

(* A continuation bound in front of the maps of the first run, and not
   named. *)
Lemma conts_related_skip (rets : finite_map (list flambda_kind))
    (ks : list continuation)
    (k : continuation) (c : cont_value) (mask : list bool) (ns : list node)
    (conts conts' : cont_env) (masks : finite_map (list bool))
    (ce : finite_map (list node)) :
  conts_related cont_rel over_rel rets ks conts conts' masks ce ->
  ~ k ∈ ks ->
  conts_related cont_rel over_rel (drop_rets [k] rets) ks ((k, c) :: conts)
    conts' ((k, mask) :: masks) ((k, ns) :: ce).
Proof.
  intros H Hk. apply (conts_related_rets rets).
  - intros j Hj. unfold resolve. simpl.
    destruct (Pos.eqb_spec j k) as [-> | _]; [ contradiction | exact (H j Hj) ].
  - intros r Hrk. rewrite lookup_drop_rets.
    destruct (among r [k]) eqn:E; [ | reflexivity ].
    apply among_In in E. destruct E as [-> | []]. contradiction.
Qed.

(* ------------------------------------------------------------------------ *)
(* Steps                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma among_flat_map_false {A : Type} (x : positive)
    (f : A -> list positive) (l : list A) (a : A) :
  among x (flat_map f l) = false -> a ∈ l -> among x (f a) = false.
Proof.
  intros H Hin. destruct (among x (f a)) eqn:E; [ | reflexivity ].
  apply among_In in E. rewrite <- H. symmetry. apply among_In.
  apply in_flat_map. exists a. auto.
Qed.

Lemma fresh_kinded_let_static (rets : finite_map (list flambda_kind))
    (masks : finite_map (list bool))
    (conts : cont_env) (bindings : list static_binding) (body : expr) :
  fresh_kinded rets masks conts (Let_static bindings body) ->
  fresh_kinded rets masks conts body.
Proof.
  intros [Hf [(ks & Hk & Ha) (xs & Hx & Hxa)]].
  cbn [continuations_of calls_kinded raises_kept] in Hf, Hk, Hx.
  rewrite among_app in Hf. apply orb_false_iff in Hf as [_ Hf].
  apply andb_true_iff in Hk as [_ Hk]. apply andb_true_iff in Hx as [_ Hx].
  split; [ exact Hf | ].
  split; [ exists ks | exists xs ]; split; assumption.
Qed.

Lemma fresh_kinded_let_cont (rets : finite_map (list flambda_kind))
    (masks : finite_map (list bool))
    (conts : cont_env) (k : continuation) (h : continuation_handler)
    (body : expr) (mask : list bool) (vars : env) :
  fresh_kinded rets masks conts (Let_cont_non_recursive k h body) ->
  (handler_is_exn_handler h = true -> hd true mask = true) ->
  handler_fresh_kinded rets masks conts h
  ∧ fresh_kinded (drop_rets [k] rets) ((k, mask) :: masks)
      ((k, Closure h vars conts []) :: conts) body.
Proof.
  intros [Hf [(ks & Hk & Ha) (xs & Hx & Hxa)]] Hmask.
  cbn [continuations_of calls_kinded raises_kept] in Hf, Hk, Hx.
  rewrite among_cons, among_app in Hf.
  apply orb_false_iff in Hf as [_ Hf]. apply orb_false_iff in Hf as [Hfh Hfb].
  apply andb_true_iff in Hk as [Hkh Hkb].
  apply andb_true_iff in Hx as [Hxh Hxb].
  split.
  { split; [ exact Hfh | ].
    split; [ exists ks | exists xs ]; split; assumption. }
  split; [ exact Hfb | split ].
  - exists ((k, handler_kinds h) :: ks).
    split; [ exact Hkb | ].
    intros j m Hj Hjr. rewrite lookup_drop_rets in Hjr.
    simpl in Hj |- *. destruct (Pos.eqb_spec j k) as [-> | Hne].
    + exists (Closure h vars conts []). split; reflexivity.
    + cbn [among existsb] in Hjr.
      rewrite (proj2 (Pos.eqb_neq _ _) Hne) in Hjr.
      exact (Ha j m Hj Hjr).
  - exists ((k, handler_is_exn_handler h) :: xs).
    split; [ exact Hxb | ].
    intros j. simpl. destruct (Pos.eqb j k); [ exact Hmask | exact (Hxa j) ].
Qed.

Lemma fresh_kinded_arm (rets : finite_map (list flambda_kind))
    (masks : finite_map (list bool)) (conts : cont_env)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (i : Z)
    (ac : apply_cont) :
  fresh_kinded rets masks conts (Switch scrutinee arms) -> (i, ac) ∈ arms ->
  fresh_kinded rets masks conts (Apply_cont ac).
Proof.
  intros [Hf [(ks & _ & Ha) (xs & _ & Hxa)]] Hin.
  split;
    [ | split; [ exists ks; split; [ reflexivity | exact Ha ]
               | exists xs; split; [ reflexivity | exact Hxa ] ] ].
  cbn [continuations_of] in Hf |- *.
  exact (among_flat_map_false _ _ _ (i, ac) Hf Hin).
Qed.


(* The values of the exits of the two runs. *)
Definition exits_related (vs vs' : list value) : Prop :=
  values_related (fun v v' => observed_equal v v' ∧ (exact = true -> v' = v))
    vs vs'.

Lemma exits_related_eq (vs vs' : list value) :
  exact = true -> exits_related vs vs' -> vs' = vs.
Proof.
  intros He. revert vs'. induction vs as [ | v vs IH]; intros [ | v' vs'] H;
    simpl in H; try contradiction; [ reflexivity | ].
  destruct H as [[_ Hv] H]. rewrite (Hv He), (IH vs' H). reflexivity.
Qed.

Lemma exits_related_observed (vs vs' : list value) :
  exits_related vs vs' -> values_related observed_equal vs vs'.
Proof.
  revert vs'. induction vs as [ | v vs IH]; intros [ | v' vs'] H;
    simpl in H; try contradiction; [ exact I | ].
  destruct H as [[Ho _] H]. split; [ exact Ho | exact (IH vs' H) ].
Qed.

Lemma map_fst_rec_env (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) :
  map fst handlers ++ map fst (env_conts en)
  = map fst (env_conts (rec_env en handlers)).
Proof.
  unfold rec_env. simpl. rewrite map_app, map_map. f_equal.
  apply map_ext. intros [k h]. reflexivity.
Qed.

(* Each construct: a step of the first run is answered by the second. *)

Ltac related_intro :=
  let Hflow := fresh "Hflow" in
  intros [Hflow (ctx & en & ce & rets & He & Ht & Henv & Hconts & Hst & Hheap
                 & Hok & Hpend)];
  pose proof (step_flow_invariant is_local G sol noted deps cds keep codes
                symbols u False no_loc Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
                Hnd Hcds _ (fun H => H) (fun _ => eq_refl) Hflow) as Hflow';
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow)
    as (Hvars & _ & Hsflow);
  simpl_state; simpl in He.

(* The variables projected next from a set, in the first run. *)
Lemma pending_let (en : rebuild_env) (vars vars' : env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  pending_ok en vars vars' (Let x kind d body) ->
  env_projections en = []
  ∨ ∃ (x0 : variable) (slot o : function_slot)
      (others : list function_slot) (slot0 : function_slot)
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * value)) (closure' : value),
      d = Project_function_slot slot o (Var x0) ∧ x <> x0
      ∧ env_projections en = x :: projected_closures x0 slot others body
      ∧ lookup x0 vars = Some (Value_closure_of_set slot0 decls vs)
      ∧ lookup x0 vars' = Some closure'
      ∧ rel (Value_closure_of_set slot0 decls vs) (Node_variable x0)
          closure'.
Proof.
  intros [Hep | (x0 & slot & others & closure & closure' & Hep & Hl & Hl'
                 & Hrel & slot0 & decls & vs & ->)];
    [ left; exact Hep | ].
  destruct others as [ | o others]; [ left; exact Hep | ].
  destruct d as [ | | | | | | mf mt [x1 | | ]]; try (left; exact Hep).
  simpl in Hep.
  destruct (Pos.eqb_spec x1 x0) as [-> | _]; [ | left; exact Hep ].
  destruct (Pos.eqb_spec mf slot) as [-> | _]; [ | left; exact Hep ].
  destruct (Pos.eqb_spec mt o) as [-> | _]; [ | left; exact Hep ].
  destruct (Pos.eqb_spec x x0) as [-> | Hne]; [ left; exact Hep | ].
  right. exists x0, slot, o, others, slot0, decls, vs, closure'.
  split; [ reflexivity | split; [ exact Hne | ] ].
  split; [ exact Hep | ].
  split; [ exact Hl | split; [ exact Hl' | exact Hrel ] ].
Qed.

(* No projection is pending before an expression that is not a [Let]. *)
Lemma pending_not_let (en : rebuild_env) (vars vars' : env) (e : expr) :
  pending_ok en vars vars' e ->
  (∀ (x : variable) (slot : function_slot) (others : list function_slot),
     projected_closures x slot others e = []) ->
  env_projections en = [].
Proof.
  intros [Hep | (x & slot & others & _ & _ & Hep & _)] Hnil; [ exact Hep | ].
  rewrite Hep. apply Hnil.
Qed.

(* The function slots of a set other than the first are among its
   slots. *)
Lemma other_slots_bound (slot : function_slot)
    (decls : list (function_slot * code_id)) (o : function_slot) :
  o ∈ other_slots slot decls -> ∃ id : code_id, lookup o decls = Some id.
Proof.
  destruct decls as [ | [first id] others]; simpl; [ intros [] | ].
  destruct (Pos.eqb first slot); [ | intros [] ].
  intros Hin. destruct (Pos.eqb o first); [ eauto | ].
  revert Hin.
  induction others as [ | [g id'] others IH]; simpl; [ intros [] | ].
  intros [-> | Hin]; [ rewrite Pos.eqb_refl; eauto | ].
  destruct (Pos.eqb o g); [ eauto | exact (IH Hin) ].
Qed.

(* The closures projected from a set next have a source: they are
   projected from a closure that flows to the variable of the set. *)
Lemma projections_live (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id)) (vs : list (value_slot * value))
    (id0 : code_id) :
  lookup slot decls = Some id0 ->
  flows_ (Value_closure_of_set slot decls vs) (Node_variable x) ->
  ∀ (others : list function_slot) (ctx : option code_id)
    (ce : finite_map (list node)) (body : expr),
    (∀ o : function_slot,
       o ∈ others -> ∃ id : code_id, lookup o decls = Some id) ->
    edges ctx ce body ->
    existsb (fun y => is_dead_var sol y Value)
      (projected_closures x slot others body)
    = false.
Proof.
  intros Hslot (sites & Hn & Hall).
  induction others as [ | o others IH]; intros ctx ce body Hothers Hedges;
    [ destruct body; reflexivity | ].
  destruct body as [y k d body | | | | | | | ]; try reflexivity.
  destruct d as [ | | | | | | mf mt [x1 | | ]]; try reflexivity.
  cbn [projected_closures].
  destruct (Pos.eqb_spec x1 x) as [-> | _]; [ | reflexivity ].
  destruct (Pos.eqb_spec mf slot) as [-> | _]; [ | reflexivity ].
  destruct (Pos.eqb_spec mt o) as [-> | _]; [ | reflexivity ].
  destruct (Pos.eqb y x); [ reflexivity | ].
  cbn [andb negb existsb].
  destruct Hedges as [((Hacc & _) & Hslots) Hbody].
  destruct (Hothers o (or_introl eq_refl)) as (id1 & Hl1).
  assert (Hfy : flows_ (Value_closure_of_set o decls vs) (Node_variable y)).
  { destruct (Hall slot id0 Hslot) as (_ & _ & Hfs & _).
    apply (flows_read is_local G sol keep called no_loc Hc _ _ _ _ _ _ (Hslots
        _)
             (Hfs o id1 Hl1) Hacc Hn).
    simpl. exists sites.
    split; [ exact (proj1 (Hall o id1 Hl1)) | exact Hall ]. }
  unfold is_dead_var.
  rewrite (flows_hs is_local G sol keep called no_loc Hc _ _ Hfy). simpl.
  exact (IH ctx ce body (fun o' Hin => Hothers o' (or_intror Hin)) Hbody).
Qed.

(* A simple of the first run, rewritten, has no location for a value in
   the second. *)
Lemma rewrite_simple_no_location (en : rebuild_env) (vars vars' : env)
    (s : simple) (v v' : value) (l : location) :
  env_ok (env_kinds en) vars vars' ->
  eval_simple vars s = Some v ->
  eval_simple vars' (rewrite_simple sol en s) = Some v' ->
  v' <> Value_location l.
Proof.
  intros Henv Hs Hs'.
  destruct s as [x | sym | c]; simpl in Hs.
  - unfold rewrite_simple in Hs'. cbv zeta in Hs'.
    rewrite (env_ok_kind en vars vars' x v Henv Hs) in Hs'.
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hu.
    + destruct (proj2 (Henv x v Hs) Hu) as (v'' & Hl & Hrel).
      simpl in Hs'. rewrite Hl in Hs'. injection Hs' as <-.
      exact (rel_not_location _ _ _ _ _ _ _ _ _ _ Hrel).
    + simpl in Hs'. injection Hs' as <-. discriminate.
  - unfold rewrite_simple in Hs'.
    destruct (is_symbol_used _ _); simpl in Hs'; injection Hs' as <-;
      discriminate.
  - simpl in Hs'. injection Hs' as <-. discriminate.
Qed.

Lemma rebuild_fields_in (en : rebuild_env) (base : node) (i : nat)
    (fields : list simple) (s' : simple) :
  s' ∈ rebuild_fields sol en base i fields ->
  (∃ c : const, s' = Const c)
  ∨ ∃ s : simple, s ∈ fields ∧ s' = rewrite_simple sol en s.
Proof.
  revert i. induction fields as [ | s fields IH]; intros i Hin;
    [ destruct Hin | ].
  cbn [rebuild_fields] in Hin. destruct Hin as [<- | Hin].
  - destruct (field_used _ _ _).
    + right. exists s. split; [ left; reflexivity | reflexivity ].
    + left. eexists. reflexivity.
  - destruct (IH (S i) Hin) as [Hcst | (s0 & Hs0 & ->)]; [ left; exact Hcst | ].
    right. exists s0. split; [ right; exact Hs0 | reflexivity ].
Qed.

(* The rebuild of a primitive is the same primitive, on constants and on
   its arguments rewritten. *)
Lemma rebuild_named_prim (en : rebuild_env) (x : variable) (p p' : prim)
    (args args' : list simple) :
  rebuild_named sol en x (Prim p args) = Prim p' args' ->
  p' = p
  ∧ ∀ s' : simple, s' ∈ args' ->
    (∃ c : const, s' = Const c)
    ∨ ∃ s : simple, s ∈ args ∧ s' = rewrite_simple sol en s.
Proof.
  intros H.
  destruct p as [[] | [] | [] | [] | []]; cbn [rebuild_named] in H;
    repeat match type of H with
           | context [if ?b then _ else _] => destruct b
           end;
    try discriminate H; injection H as <- <-;
    (split; [ reflexivity | intros s' Hin ]);
    first
      [ apply in_map_iff in Hin as (s & <- & Hs); right; exists s;
        split; [ exact Hs | reflexivity ]
      | eapply rebuild_fields_in; exact Hin ].
Qed.

(* The step of a [Let] of the second run that is the rebuild of one of
   the first, which is of no primitive of the heap: by [eval_named], as
   neither has a location among the values of its arguments. *)
Lemma rebuilt_let_step (en : rebuild_env) (vars vars' : env)
    (x y : variable) (kind : flambda_kind) (d : named) (body' : expr)
    (conts' : cont_env) (statics' : static_env) (hp : heap)
    (rcd : read_record) :
  env_ok (env_kinds en) vars vars' ->
  (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
  (∀ (p : prim) (args : list simple),
     d = Prim p args -> ∃ vs : list value, eval_simples vars args = Some vs) ->
  step (State (Let y kind (rebuild_named sol en x d) body') vars' conts'
          statics' hp rcd)
  = match eval_named statics' vars' (rebuild_named sol en x d) with
    | Some v =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body' ((y, v) :: vars') conts' statics' hp rcd)
      else Undefined
    | None => Undefined
    end.
Proof.
  intros Henv Hnoheap Hsrc. apply step_let_eval.
  - intros q args' H.
    destruct d as [s | p args | | | | | ]; cbn [rebuild_named] in H;
      try discriminate H.
    destruct (rebuild_named_prim _ _ _ _ _ _ H) as [Hp _]. subst p.
    exact (Hnoheap q args eq_refl).
  - intros p' args' vs' l H Hvs' Hin.
    destruct d as [s | p args | | | | | ]; cbn [rebuild_named] in H;
      try discriminate H.
    destruct (rebuild_named_prim _ _ _ _ _ _ H) as [_ Hargs].
    destruct (Hsrc p args eq_refl) as (vs & Hvs).
    destruct (eval_simples_in _ _ _ _ Hvs' Hin) as (s' & Hs'in & Hs').
    destruct (Hargs s' Hs'in) as [(c & ->) | (s & Hs & ->)].
    + simpl in Hs'. discriminate Hs'.
    + destruct (eval_simples_member _ _ _ _ Hvs Hs) as (w & Hw).
      exact (rewrite_simple_no_location en vars vars' s w _ l Henv Hw Hs'
               eq_refl).
Qed.

Lemma project_function_slot_not_prim (a b : function_slot) (s : simple) :
  ∀ (p : prim) (args : list simple), Project_function_slot a b s <> Prim p args.
Proof. intros p args H. discriminate H. Qed.

Lemma closure_of_set_not_prim (slot : function_slot)
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) :
  ∀ (p : prim) (args : list simple),
    Closure_of_set slot decls value_slots <> Prim p args.
Proof. intros p args H. discriminate H. Qed.

Lemma let_answered (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Let x kind d body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Let x kind d body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  related_intro. destruct He as [Hd Hbody_edges].
  destruct Hok as (Hlive & Hexits & Hscr & Hcf & Hkd & Hnr).
  simpl in Hexits, Hscr, Hcf, Hnr.
  apply andb_true_iff in Hcf as [Hfrag Hcf_body].
  pose proof (conj Hlive (conj Hexits (conj Hscr (conj Hcf_body
                (conj Hkd Hnr)))))
    as Hok_b.
  assert (Hnoheap : ∀ (q : heap_primitive) (args : list simple),
             d <> Prim (Heap q) args)
    by (eapply named_edges_not_heap; exact Hd).
  assert (Hnoloc : ∀ (p : prim) (args : list simple) (vs : list value)
                     (l : location),
             d = Prim p args -> eval_simples vars args = Some vs ->
             Value_location l ∉ vs)
    by (intros p args vs l _ Hvs; eapply env_flows_no_location;
        [ reflexivity | exact Hvars | exact Hvs ]).
  unfold answered.
  rewrite (step_let_eval _ _ _ _ _ _ _ _ _ Hnoheap Hnoloc) in Hflow' |- *.
  destruct (eval_named statics vars d) as [v | ] eqn:Hev; [ | exact I ].
  assert (Hsrc : ∀ (p : prim) (args : list simple),
             d = Prim p args ->
             ∃ vs : list value, eval_simples vars args = Some vs).
  { intros p args ->. cbn [eval_named] in Hev.
    destruct (eval_simples vars args) as [vs | ];
      [ eexists; reflexivity | discriminate Hev ]. }
  destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | exact I ].
  apply kind_eqb_eq in Hkind. subst kind.
  assert (Hflows : flows_ v (Node_variable x)).
  { destruct (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
      as [Hvars' _].
    apply Hvars'. simpl. rewrite Pos.eqb_refl. reflexivity. }
  assert (Hnotdead : is_dead_var sol x (kind_of_value v) = false).
  { pose proof (flows_hs is_local G sol keep called no_loc Hc _ _ Hflows) as
      Hhs.
    unfold is_dead_var. rewrite Hhs.
    destruct (kind_of_value v); reflexivity. }
  destruct (pending_let en vars vars' x (kind_of_value v) d body Hpend)
    as [Hep | (x0 & slot & o & others & slot0 & decls0 & vs0 & c' & -> & Hne
               & Hep & Hx0 & Hx0' & Hrel0)].
  2: { (* The next projection pending: kept as it is. *)
    cbn [eval_named eval_simple] in Hev. rewrite Hx0 in Hev.
    unfold project_function_slot in Hev. simpl in Hev.
    destruct (Pos.eqb_spec slot0 slot) as [-> | _]; [ | discriminate Hev ].
    destruct (lookup slot decls0) as [id0 | ] eqn:Hl0; [ | discriminate Hev ].
    destruct (lookup o decls0) as [id1 | ] eqn:Hl1; [ | discriminate Hev ].
    injection Hev as <-.
    pose proof Hrel0 as Hrel1. cbn [rel] in Hrel1.
    destruct Hrel1 as (sites & vs' & _ & -> & _).
    destruct Hd as ((Hacc & _) & Hslots).
    pose proof (rel_project_function_slot slot o decls0 vs0 vs' _ _ id0 id1
                  (Hslots _) Hl0 Hl1 Hacc Hrel0) as Hrel_x.
    assert (Hev' : eval_named statics' vars'
                     (Project_function_slot slot o (Var x0))
                   = Some (Value_closure_of_set o decls0 vs')).
    { cbn [eval_named eval_simple]. rewrite Hx0'.
      unfold project_function_slot. simpl.
      rewrite Pos.eqb_refl, Hl0, Hl1. reflexivity. }
    revert Ht.
    rewrite (rebuild_let_pending_eq en x _ slot o (Var x0) body _ Hep).
    intros ->. right. left. eexists. split.
    - rewrite (step_let_not_prim _ _ _ _ _ _ _ _ _
                 (project_function_slot_not_prim _ _ _)), Hev'.
      simpl kind_of_value. rewrite kind_eqb_refl.
      reflexivity.
    - split; [ exact Hflow' | ].
      eexists ctx, _, ce, rets. simpl_state.
      split; [ exact Hbody_edges | split; [ reflexivity | ] ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
      + apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
        * intros _. eexists. split; [ reflexivity | exact Hrel_x ].
        * right. eexists. reflexivity.
      + split; [ exact Hheap | ].
        split; [ exact Hok_b | ].
        assert (Hne' : Pos.eqb x0 x = false)
          by (apply Pos.eqb_neq; intros ->; exact (Hne eq_refl)).
        right. exists x0, slot, others, (Value_closure_of_set slot decls0 vs0),
          (Value_closure_of_set slot decls0 vs').
        split; [ reflexivity | ]. simpl. rewrite Hne'.
        split; [ exact Hx0 | split; [ exact Hx0' | split; [ exact Hrel0 | ] ] ].
        eauto. }
  destruct (named_plain d) eqn:Hplain.
  - revert Ht. rewrite (rebuild_let_eq en x _ d body Hep Hplain).
    destruct (is_dead_var sol x (kind_of_value v)) eqn:Hdead;
      [ congruence | ].
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hused; intros ->.
    + right. left.
      destruct (named_rel en vars vars' statics statics' x d v Henv Hst Hvars
                  Hsflow Hfrag Hplain Hd Hev Hflows Hused)
        as (v' & Hev' & Hrel).
      eexists. split.
      * rewrite (rebuilt_let_step en vars vars' x x _ d _ _ _ _ _ Henv Hnoheap
                   Hsrc),
          Hev', (rel_kind _ _ _ _ _ _ _ _ _ Hrel),
          kind_eqb_refl.
        reflexivity.
      * split; [ exact Hflow' | ].
        exists ctx, (bind_variables en [(x, kind_of_value v)]), ce, rets.
        simpl_state.
        split; [ exact Hbody_edges | split; [ reflexivity | ] ].
        split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
        -- apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
           ++ intros _. exists v'. auto.
           ++ right. exists v'. reflexivity.
        -- split; [ exact Hheap | ].
           split; [ exact Hok_b | ].
           left. reflexivity.
    + right. right. left. split; [ unfold measure; simpl; lia | ].
      split; [ exact Hflow' | ].
      exists ctx, (bind_variables en [(x, kind_of_value v)]), ce, rets.
      simpl_state.
      split; [ exact Hbody_edges | split; [ reflexivity | ] ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
      * apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
        -- intros Hu. congruence.
        -- left. reflexivity.
      * split; [ exact Hheap | ].
        split; [ exact Hok_b | ].
        left. reflexivity.
  - (* A closure of a set. *)
    destruct d as [ | | code vs | slot decls value_slots | | | ];
      try discriminate Hplain; simpl in Hfrag;
      [ unfold named_checked in Hfrag; destruct exact; discriminate Hfrag | ].
    simpl in Hev.
    destruct (lookup slot decls) as [id0 | ] eqn:Hid; [ | discriminate Hev ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    injection Hev as <-.
    set (bound := set_closure_nodes F (Node_variable x)).
    destruct (value_slots_rel en vars vars' bound value_slots vs Henv Hvs)
      as (vs' & Hvs' & Hempty & Hslots_rel).
    assert (Hev' : eval_named statics' vars'
                     (Closure_of_set slot decls
                        (rebuild_value_slots sol en bound value_slots))
                   = Some (Value_closure_of_set slot decls vs')).
    { simpl. rewrite Hid, Hvs'. reflexivity. }
    destruct (Hd id0 Hid) as (sites & Hsx & Hsites).
    assert (Hrel_x : rel (Value_closure_of_set slot decls vs) (Node_variable x)
                       (Value_closure_of_set slot decls vs')).
    { cbn [rel]. exists sites, vs'.
      assert (Hall : ∀ (f : function_slot) (id : code_id),
                lookup f decls = Some id ->
                may_come_from sol (sites f) (sites f)
                ∧ called (sites f) id
                ∧ (∀ (g : function_slot) (id' : code_id),
                     lookup g decls = Some id' ->
                     g_constructor G (sites f) (Field_function_slot g)
                       (sites g))
                ∧ slots_related (slot_rel (sites f) vs') vs []).
      { intros f id Hf. destruct (Hsites f id Hf) as (Hdep & Hfs & Hws).
        split; [ exact (called_alloc is_local G sol cds Hc Hcds _ _
                          (Hdeps _ Hdep)) | ].
        split; [ exact (Hdeps _ Hdep) | ].
        split; [ exact Hfs | ].
        apply (Hslots_rel (sites f)); [ | exact Hws ].
        destruct (Hsites slot id0 Hid) as (_ & Hfs0 & _).
        pose proof (Hfs0 f id Hf) as Hcons. rewrite Hsx in Hcons.
        rewrite HG in Hcons. simpl in Hcons.
        unfold bound, set_closure_nodes. right.
        apply in_flat_map.
        exists (Node_variable x, Field_function_slot f, sites f).
        split; [ exact Hcons | ]. simpl. rewrite node_eqb_refl. left.
        reflexivity. }
      split; [ rewrite <- Hsx; exact (proj1 (Hall slot id0 Hid)) | ].
      split; [ reflexivity | split; [ | exact Hall ] ].
      intros He. apply Hempty. unfold named_checked in Hfrag.
      rewrite He in Hfrag. simpl in Hfrag.
      destruct value_slots; [ reflexivity | discriminate Hfrag ]. }
    pose proof (projections_live x slot decls vs id0 Hid Hflows
                  (other_slots slot decls) ctx ce body
                  (other_slots_bound slot decls) Hbody_edges) as Hlive_p.
    revert Ht.
    rewrite (rebuild_let_set_eq en x _ slot decls value_slots body Hep).
    fold bound.
    match goal with
    | |- context [if ?b then Invalid else _] => destruct b eqn:Hdead
    end;
      [ rewrite Hnotdead, Hlive_p in Hdead; discriminate Hdead | ].
    match goal with
    | |- context [if ?b then Let _ _ _ _ else _] => destruct b eqn:Hused
    end;
      intros ->.
    + right. left. eexists. split.
      * rewrite (step_let_not_prim _ _ _ _ _ _ _ _ _
                   (closure_of_set_not_prim _ _ _)), Hev', kind_eqb_refl.
        reflexivity.
      * split; [ exact Hflow' | ].
        eexists ctx, _, ce, rets. simpl_state.
        split; [ exact Hbody_edges | split; [ reflexivity | ] ].
        split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
        -- apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
           ++ intros _. eexists. split; [ reflexivity | exact Hrel_x ].
           ++ right. eexists. reflexivity.
        -- split; [ exact Hheap | ].
           split; [ exact Hok_b | ].
           right. exists x, slot, (other_slots slot decls),
             (Value_closure_of_set slot decls vs),
             (Value_closure_of_set slot decls vs').
           split; [ reflexivity | ]. simpl. rewrite Pos.eqb_refl.
           split; [ reflexivity | split; [ reflexivity | ] ].
           split; [ exact Hrel_x | eauto ].
    + right. right. left. split; [ unfold measure; simpl; lia | ].
      split; [ exact Hflow' | ].
      eexists ctx, _, ce, rets. simpl_state.
      split; [ exact Hbody_edges | split; [ reflexivity | ] ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
      * apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
        -- intros Hu. apply orb_false_iff in Hused as [Hu' _]. congruence.
        -- left. reflexivity.
      * split; [ exact Hheap | ].
        split; [ exact Hok_b | ].
        left. reflexivity.
Qed.

Lemma let_static_answered (bindings : list static_binding) (body : expr)
    (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Let_static bindings body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Let_static bindings body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  related_intro. destruct He as [Hbindings Hbody_edges].
  destruct Hok as (Hlive & Hexits & Hscr & Hcf & Hkd & Hnr).
  simpl in Hexits, Hscr, Hcf, Hnr. apply fresh_kinded_let_static in Hkd.
  destruct Hscr as [_ Hscr].
  apply andb_true_iff in Hcf as [Hfrag Hcf_body].
  pose proof (conj Hlive (conj Hexits (conj Hscr (conj Hcf_body
                (conj Hkd Hnr)))))
    as Hok_b.
  unfold answered. rewrite step_let_static in Hflow' |- *.
  destruct (eval_static_bindings vars bindings statics) as [statics2 | ]
    eqn:Hev; [ | exact I ].
  destruct (static_bindings_rel en vars vars' Henv bindings statics statics'
              statics2 Hfrag Hbindings Hst Hev)
    as (statics2' & Hev' & Hst2).
  rewrite rebuild_let_static_eq in Ht. subst e'.
  right. left. eexists. split.
  - rewrite step_let_static, Hev'. reflexivity.
  - split; [ exact Hflow' | ].
    exists ctx, (bind_variables en []), ce, rets. simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split.
    + apply (conts_related_sub _ _ _ _ _ _ _ Hconts). intros k Hk. simpl.
      apply in_or_app. right. exact Hk.
    + split; [ exact Hst2 | split; [ exact Hheap | ] ].
      split; [ exact Hok_b | left;
        reflexivity ].
Qed.

Lemma bound_ok_app (rets : finite_map (list flambda_kind)) (en : rebuild_env)
    (bound : list continuation) (bs : list (continuation * list bool)) :
  bound_ok rets en bound ->
  bound_ok (drop_rets (map fst bs) rets) (bind_continuations en bs)
    (map fst bs ++ bound).
Proof.
  intros [Hsub Hrets]. split.
  - intros j Hj. cbn [env_conts bind_continuations].
    rewrite map_app, among_app in *.
    apply orb_true_iff in Hj as [Hj | Hj]; apply orb_true_iff;
      [ left; exact Hj | right; exact (Hsub j Hj) ].
  - intros r K Hr. apply lookup_drop_rets_some in Hr as [Hr Hrk].
    rewrite among_app, Hrk, (Hrets r K Hr). reflexivity.
Qed.

Lemma let_cont_non_recursive_answered (k : continuation)
    (h : continuation_handler) (body : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Let_cont_non_recursive k h body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Let_cont_non_recursive k h body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  related_intro. destruct He as [Hh_edges Hbody_edges].
  unfold answered. rewrite step_let_cont_non_recursive in Hflow' |- *.
  destruct Hok as (Hlive & (bound & Hbound & Hexits) & Hscr & Hcf
                   & Hkd & Hnr).
  simpl in Hexits, Hscr, Hcf, Hnr.
  apply andb_true_iff in Hexits as [Hexits_h Hexits_body].
  destruct Hscr as [Hscr_h Hscr_body].
  apply andb_true_iff in Hcf as [Hcf_h Hcf_body].
  set (mask := handler_mask sol (handler_is_exn_handler h) h) in *.
  destruct (fresh_kinded_let_cont _ _ _ k h body mask vars Hkd
              (handler_mask_exn_first h))
    as [Hkd_h Hkd_body].
  set (en' := bind_continuations en [(k, mask)]).
  assert (Hok_h : handler_ok ctx rets en ce conts h).
  { split; [ exact Hlive | ].
    split; [ exists bound; split; [ exact Hbound | exact Hexits_h ] | ].
    split; [ exact Hscr_h | split; [ exact Hcf_h | split; [ exact Hkd_h | ] ] ].
    intros r K Hr. destruct (Hnr r K Hr) as [Hr1 Hr2].
    apply andb_true_iff in Hr1 as [Hr1 _]. split; [ exact Hr1 | ].
    apply andb_true_iff in Hr2. exact (proj1 Hr2). }
  assert (Hok_body : expr_ok ctx (drop_rets [k] rets) en'
                       ((k, handler_params h) :: ce)
                       ((k, Closure h vars conts []) :: conts) body).
  { split; [ exact Hlive | ].
    split; [ exists (k :: bound); split;
             [ exact (bound_ok_app rets en bound [(k, mask)] Hbound)
             | exact Hexits_body ] | ].
    split; [ exact Hscr_body | split; [ exact Hcf_body |
      split; [ exact Hkd_body | ] ] ].
    intros r K Hr. apply lookup_drop_rets_some in Hr as [Hr Hrk].
    rewrite among_cons in Hrk. apply orb_false_iff in Hrk as [Hrk _].
    destruct (Hnr r K Hr) as [Hr1 Hr2].
    apply andb_true_iff in Hr1 as [_ Hr1]. rewrite Pos.eqb_sym, Hrk in Hr1.
    split; [ exact Hr1 | ].
    apply andb_true_iff in Hr2 as [_ H]. rewrite Pos.eqb_sym, Hrk in H.
    exact H. }
  rewrite rebuild_let_cont_non_recursive_eq in Ht. fold mask en' in Ht.
  destruct (among k (free_continuations_of_expr (rebuild en' body))) eqn:Hk;
    subst e'.
  - right. left. eexists. split; [ rewrite step_let_cont_non_recursive;
                                    reflexivity | ].
    split; [ exact Hflow' | ].
    exists ctx, en', ((k, handler_params h) :: ce), (drop_rets [k] rets).
    simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
    + intros j Hj. simpl in Hconts. unfold resolve. simpl.
      destruct (Pos.eqb_spec j k) as [-> | Hne].
      * right. left. eexists _, _, _. split; [ reflexivity | ].
        split; [ reflexivity | ]. split; [ reflexivity | ].
        split; [ reflexivity | ].
        split; [ | apply lookup_drop_rets_none; right;
                   rewrite among_cons, Pos.eqb_refl; reflexivity ].
        apply (Cont_rel_non_recursive _ _ _ _ _ _ ctx ce rets).
        -- exact Henv.
        -- exact Hok_h.
        -- exact Hh_edges.
        -- apply (conts_related_sub _ _ _ _ _ _ _ Hconts). intros j' Hj'.
           apply in_or_app. left. exact Hj'.
      * assert (Hj' : j ∈ free_continuations_of_expr
                            (Let_cont_non_recursive k
                               (rebuild_h en true mask h) (rebuild en' body))).
        { apply in_or_app. right.
          apply among_In. rewrite among_without. apply andb_true_intro.
          split; [ apply among_In; exact Hj | ].
          simpl. rewrite (proj2 (Pos.eqb_neq j k) Hne). reflexivity. }
        destruct (Hconts j Hj')
          as [A | [(mask0 & c0 & c0' & B) | (K & HK & C)]];
          [ left; exact A
          | right; left; exists mask0, c0, c0';
            exact (conts_rel_handler_drop _ _ _ _ _ _ _ _ _ _ B) | ].
        right. right. exists K. split; [ | exact C ].
        rewrite lookup_drop_rets, among_cons,
          (proj2 (Pos.eqb_neq j k) Hne).
        exact HK.
    + split; [ exact Hok_body | ].
      left. reflexivity.
  - right. right. left. split; [ unfold measure; simpl; lia | ].
    split; [ exact Hflow' | ].
    exists ctx, en', ((k, handler_params h) :: ce), (drop_rets [k] rets).
    simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
    + apply conts_related_skip; [ exact Hconts | ].
      intros Hin. apply among_In in Hin. congruence.
    + split; [ exact Hok_body | ].
      left. reflexivity.
Qed.

Lemma lookup_rebuilt_handlers (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  lookup k (rebuilt_handlers en handlers)
  = match lookup k handlers with
    | Some h => Some (rebuild_h en false (handler_mask sol false h) h)
    | None => None
    end.
Proof.
  unfold rebuilt_handlers.
  exact (lookup_map_pairs
           (fun h => rebuild_h en false (handler_mask sol false h) h) k
           handlers).
Qed.

Lemma lookup_rec_env (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  lookup k (env_conts (rec_env en handlers))
  = match lookup k handlers with
    | Some h => Some (handler_mask sol false h)
    | None => lookup k (env_conts en)
    end.
Proof.
  unfold rec_env, bind_continuations. cbn [env_conts].
  exact (lookup_map_pairs_app (fun h => handler_mask sol false h) k handlers
           (env_conts en)).
Qed.

Lemma lookup_rec_conts (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  lookup k (rec_conts ce handlers)
  = match lookup k handlers with
    | Some h => Some (handler_params h)
    | None => lookup k ce
    end.
Proof.
  unfold rec_conts. exact (lookup_map_pairs_app handler_params k handlers ce).
Qed.

Lemma fresh_kinded_let_cont_rec (rets : finite_map (list flambda_kind))
    (en : rebuild_env) (conts : cont_env)
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (vars : env) :
  fresh_kinded rets (env_conts en) conts (Let_cont_recursive handlers body) ->
  (∀ (k : continuation) (h : continuation_handler),
     (k, h) ∈ handlers ->
     handler_fresh_kinded (drop_rets (map fst handlers) rets)
       (env_conts (rec_env en handlers))
       (rec_closures handlers vars conts ++ conts) h)
  ∧ fresh_kinded (drop_rets (map fst handlers) rets)
      (env_conts (rec_env en handlers))
      (rec_closures handlers vars conts ++ conts) body.
Proof.
  intros [Hf [(ks & Hk & Ha) (xs & Hx & Hxa)]].
  cbn [continuations_of calls_kinded raises_kept] in Hf, Hk, Hx.
  rewrite among_app in Hf. apply orb_false_iff in Hf as [Hfh Hfb].
  apply andb_true_iff in Hk as [Hkh Hkb]. rewrite forallb_forall in Hkh.
  apply andb_true_iff in Hx as [Hxh Hxb]. rewrite forallb_forall in Hxh.
  set (xs' := map (fun '(k, _) => (k, false)) handlers ++ xs) in *.
  assert (Hxa' : exn_agree (env_conts (rec_env en handlers)) xs').
  { intros j. rewrite lookup_rec_env. unfold xs'. rewrite lookup_rec_false.
    destruct (lookup j handlers) as [h | ];
      [ intros E; discriminate E | exact (Hxa j) ]. }
  set (ks' := map (fun '(k, h) => (k, handler_kinds h)) handlers ++ ks).
  assert (Ha' : kinds_agree (drop_rets (map fst handlers) rets)
                  (env_conts (rec_env en handlers)) ks'
                  (rec_closures handlers vars conts ++ conts)).
  { intros j m Hj Hjr. rewrite lookup_rec_env in Hj.
    rewrite lookup_rec_closures.
    unfold ks'. rewrite (lookup_map_pairs_app handler_kinds).
    destruct (lookup j handlers) as [h | ] eqn:Hl.
    - exists (Closure h vars conts handlers). split; reflexivity.
    - rewrite lookup_drop_rets in Hjr.
      replace (among j (map fst handlers)) with false in Hjr
        by (symmetry; exact (lookup_none_among j handlers Hl)).
      exact (Ha j m Hj Hjr). }
  split.
  - intros k h Hin. split.
    + pose proof (among_flat_map_false _ _ _ (k, h) Hfh Hin) as Hkh'.
      cbn [fst snd] in Hkh'. rewrite among_cons in Hkh'.
      apply orb_false_iff in Hkh' as [_ Hkh']. exact Hkh'.
    + split; [ exists ks'; split; [ exact (Hkh (k, h) Hin) | exact Ha' ]
             | exists xs'; split; [ exact (Hxh (k, h) Hin) | exact Hxa' ] ].
  - split; [ exact Hfb | ].
    split; [ exists ks'; split; [ exact Hkb | exact Ha' ]
           | exists xs'; split; [ exact Hxb | exact Hxa' ] ].
Qed.

(* The continuations of a recursive group and of what is around it, in
   the two runs. *)
Lemma rec_conts_related (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) (ctx : option code_id)
    (ce : finite_map (list node)) (rets : finite_map (list flambda_kind))
    (ks : list continuation) :
  env_ok (env_kinds en) vars vars' ->
  (∀ (k' : continuation) (h' : continuation_handler),
     (k', h') ∈ handlers ->
     handler_ok ctx (drop_rets (map fst handlers) rets) (rec_env en handlers)
       (rec_conts ce handlers) (rec_closures handlers vars conts ++ conts) h'
     ∧ handler_edges_ ctx (rec_conts ce handlers) h'
     ∧ conts_related cont_rel over_rel rets
         (without (map fst handlers)
            (free_continuations_of_handler
               (rebuild_h (rec_env en handlers) false
                  (handler_mask sol false h') h')))
         conts conts' (env_conts en) ce) ->
  conts_related cont_rel over_rel rets (without (map fst handlers) ks) conts
    conts' (env_conts en) ce ->
  conts_related cont_rel over_rel (drop_rets (map fst handlers) rets) ks
    (rec_closures handlers vars conts ++ conts)
    (rec_closures (rebuilt_handlers (rec_env en handlers) handlers) vars'
       conts' ++ conts')
    (env_conts (rec_env en handlers)) (rec_conts ce handlers).
Proof.
  intros Henv Hall Hks j Hj. unfold resolve.
  rewrite lookup_rec_closures, lookup_rec_closures, lookup_rebuilt_handlers,
    lookup_rec_env, lookup_rec_conts.
  destruct (lookup j handlers) as [hj | ] eqn:Hlj.
  - right. left. eexists _, _, _. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    split; [ exact (Cont_rel_recursive j hj en vars vars' conts conts'
                      handlers ctx ce rets Henv (lookup_in j hj handlers Hlj)
                      Hall) | ].
    apply lookup_drop_rets_none. right. apply among_In.
    apply (in_map fst _ (j, hj)). exact (lookup_in j hj handlers Hlj).
  - assert (Hj' : j ∈ without (map fst handlers) ks).
    { apply among_In. rewrite among_without.
      replace (among j (map fst handlers)) with false
        by (symmetry; exact (lookup_none_among j handlers Hlj)).
      simpl. rewrite andb_true_r. apply among_In. exact Hj. }
    destruct (Hks j Hj') as [A | [(mask0 & c0 & c0' & B) | (K & HK & C)]];
      [ left; exact A
      | right; left; exists mask0, c0, c0';
        exact (conts_rel_handler_drop _ _ _ _ _ _ _ _ _ _ B) | ].
    right. right. exists K. split; [ | exact C ].
    rewrite lookup_drop_rets.
    replace (among j (map fst handlers)) with false
      by (symmetry; exact (lookup_none_among j handlers Hlj)).
    exact HK.
Qed.

Lemma let_cont_recursive_answered
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Let_cont_recursive handlers body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Let_cont_recursive handlers body) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  related_intro. destruct He as [Hhandlers Hbody_edges].
  unfold answered. rewrite step_let_cont_recursive in Hflow' |- *.
  destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
  destruct Hok as (Hlive & (bound & Hbound & Hexits) & Hscr & Hcf
                   & Hkd & Hnr).
  simpl in Hexits, Hscr, Hcf, Hnr.
  destruct (fresh_kinded_let_cont_rec rets en conts handlers body vars Hkd)
    as [Hkd_h Hkd_body].
  apply andb_true_iff in Hexits as [Hexits_h Hexits_body].
  destruct Hscr as [Hscr_h Hscr_body].
  apply andb_true_iff in Hcf as [Hcf_h Hcf_body].
  rewrite forallb_forall in Hexits_h, Hcf_h.
  pose proof (bound_ok_app rets en bound
                (map (fun '(k, h) => (k, handler_mask sol false h)) handlers)
                Hbound) as Hbound'.
  replace (map fst (map (fun '(k, h) => (k, handler_mask sol false h))
                      handlers))
    with (map fst handlers) in Hbound'
    by (rewrite map_map; apply map_ext; intros [k h]; reflexivity).
  fold (rec_env en handlers) in Hbound'.
  assert (Hnr' : ∀ (r : continuation) (K : list flambda_kind),
            lookup r (drop_rets (map fst handlers) rets) = Some K ->
            (∀ (k' : continuation) (h' : continuation_handler),
               (k', h') ∈ handlers -> handler_no_raise_to r h' = true)
            ∧ no_raise_to r body = true
            ∧ (∀ (k' : continuation) (h' : continuation_handler),
                 (k', h') ∈ handlers ->
                 handler_ret_calls_ok (env_kinds en) r K h'
                 = true)
            ∧ rcalls_ok (env_kinds en) r K body = true).
  { intros r K Hr. apply lookup_drop_rets_some in Hr as [Hr Hrk].
    destruct (Hnr r K Hr) as [Hr1 Hr2].
    rewrite Hrk in Hr1. cbn [orb] in Hr1.
    apply andb_true_iff in Hr1 as [Hr1 Hr1b].
    rewrite forallb_forall in Hr1.
    split; [ | split; [ exact Hr1b | ] ].
    - intros k' h' Hin. exact (Hr1 (k', h') Hin).
    - pose proof Hr2 as H. rewrite Hrk in H. cbn [orb] in H.
      apply andb_true_iff in H as [H1 H2]. rewrite forallb_forall in H1.
      split; [ | exact H2 ]. intros k' h' Hin. exact (H1 (k', h') Hin). }
  rewrite rebuild_let_cont_recursive_eq in Ht. subst e'.
  set (en' := rec_env en handlers) in *.
  right. left. eexists. split.
  - rewrite step_let_cont_recursive.
    assert (Hbinds' : handlers_bind_statics (rebuilt_handlers en' handlers)
                      = false).
    { apply not_true_iff_false. intros Hb. unfold handlers_bind_statics in Hb.
      apply existsb_exists in Hb as ([k h'] & Hin & Hb).
      unfold rebuilt_handlers in Hin.
      apply in_map_iff in Hin as ([k' h] & [= <- <-] & Hin).
      pose proof (proj2 binds_statics_rebuild h (Hcf_h (k', h) Hin) en' false
                    (handler_mask sol false h) Hb) as Hb'.
      unfold handlers_bind_statics in Hbinds.
      rewrite (proj2 (existsb_exists _ _) (ex_intro _ (k', h) (conj Hin Hb')))
        in Hbinds.
      discriminate Hbinds. }
    rewrite Hbinds'. reflexivity.
  - split; [ exact Hflow' | ].
    exists ctx, en', (rec_conts ce handlers),
      (drop_rets (map fst handlers) rets).
    simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
    + simpl in Hconts. rewrite map_fst_rebuilt_handlers in Hconts.
      apply (rec_conts_related _ _ _ _ _ _ ctx _ rets); [ exact Henv | | ].
      * intros k' h' Hin. split; [ | split ].
        -- split; [ exact Hlive | ].
           split; [ exists (map fst handlers ++ bound);
                    split; [ exact Hbound' | exact (Hexits_h (k', h') Hin) ]
                  | ].
           split; [ exact (proj1 (every_in _ _ _) Hscr_h (k', h') Hin) | ].
           split; [ exact (Hcf_h (k', h') Hin) | ].
           split; [ exact (Hkd_h k' h' Hin) | ].
           intros r K Hr. destruct (Hnr' r K Hr) as (H1 & _ & H3 & _).
           split; [ exact (H1 k' h' Hin) | exact (H3 k' h' Hin) ].
        -- exact (proj1 (every_in _ _ _) Hhandlers (k', h') Hin).
        -- apply (conts_related_sub _ _ _ _ _ _ _ Hconts). intros j Hj.
           apply among_In. apply among_In in Hj.
           rewrite among_without in Hj |- *.
           apply andb_true_iff in Hj as [Hj Hnot].
           rewrite Hnot, andb_true_r. apply among_In. apply in_or_app. left.
           apply in_flat_map.
           exists (k', rebuild_h en' false (handler_mask sol false h') h').
           split; [ | apply among_In; exact Hj ].
           unfold rebuilt_handlers. apply in_map_iff. exists (k', h').
           auto.
      * apply (conts_related_sub _ _ _ _ _ _ _ Hconts). intros j Hj.
        apply among_In. apply among_In in Hj.
        rewrite among_without in Hj |- *.
        apply andb_true_iff in Hj as [Hj Hnot].
        rewrite Hnot, andb_true_r. apply among_In. apply in_or_app. right.
        apply among_In. exact Hj.
    + split; [ | left; reflexivity ].
      split; [ exact Hlive | ].
      split; [ exists (map fst handlers ++ bound);
               split; [ exact Hbound' | exact Hexits_body ] | ].
      split; [ exact Hscr_body | split; [ exact Hcf_body | ] ].
      split; [ exact Hkd_body | ].
      intros r K Hr. destruct (Hnr' r K Hr) as (_ & H2 & _ & H4).
      split; [ exact H2 | exact H4 ].
Qed.

(* A continuation of the unit has one parameter, at a node with any
   usage. *)
Lemma unit_conts_lookup (k : continuation) :
  unit_cont k ->
  ∃ p : node, lookup k (unit_conts u) = Some [p] ∧ p ∈ unit_cont_nodes u k.
Proof.
  intros Hk. unfold unit_cont_nodes.
  destruct (lookup k (unit_conts u)) as [ps | ] eqn:Hl.
  - unfold unit_conts in Hl. simpl in Hl.
    destruct (Pos.eqb k (unit_exn u)).
    + injection Hl as <-. exists (Node_made 2).
      split; [ reflexivity | left; reflexivity ].
    + destruct (Pos.eqb k (unit_return u)); [ | discriminate Hl ].
      injection Hl as <-. exists (Node_made 1).
      split; [ reflexivity | left; reflexivity ].
  - exfalso. unfold unit_conts in Hl. simpl in Hl.
    destruct Hk as [-> | ->]; rewrite Pos.eqb_refl in Hl;
      [ destruct (Pos.eqb (unit_return u) (unit_exn u)) | ]; discriminate Hl.
Qed.

(* The arguments of an [Apply_cont], at the nodes [ns] that [ce] has for
   its continuation, where these have a use or the values are of kind
   [Region] or [Rec_info]. *)
Lemma apply_cont_args_at (en : rebuild_env) (vars vars' : env)
    (ce : finite_map (list node)) (ac : apply_cont) (vs vs' : list value) :
  env_ok (env_kinds en) vars vars' ->
  apply_cont_edges G noted ce ac ->
  eval_simples vars (ac_args ac) = Some vs ->
  (∀ (i : nat) (s : simple) (v v' : value),
     nth_error (ac_args ac) i = Some s -> nth_error vs i = Some v ->
     nth_error vs' i = Some v' -> kept_simple en s = true ->
     rel v (node_of_simple s) v') ->
  ∀ (ns : list node) (i : nat) (n : node) (v v' : value),
    lookup (ac_continuation ac) ce = Some ns ->
    nth_error ns i = Some n -> nth_error vs i = Some v ->
    nth_error vs' i = Some v' ->
    hu n ∨ kind_of_value v = Region ∨ kind_of_value v = Rec_info ->
    rel v n v'.
Proof.
  intros Henv [Haliases _] Hvs Hrels ns i n v v' Hce Hn Hv Hv' Hu.
  destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
  assert (Hal : s_alias sol n (node_of_simple s) = true).
  { apply (alias_input _ _ _ Hc). apply (Haliases _ Hce).
    exact (nth_error_combine _ _ i _ _ Hn Hs). }
  apply (rel_alias G sol keep called (exact = true) True is_local Hc v _ _ v'
      Hal).
  apply (Hrels i s v v' Hs Hv Hv').
  destruct Hu as [Hhu | Hk].
  - apply kept_of_hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
  - exact (kept_of_region en vars vars' s v Henv Hsv Hk).
Qed.

(* Entering a handler of the first run, related to [c'']: the second
   enters [c''] at the values kept, in a related state. *)
(* The body of a handler entered, which has as many kinds as
   parameters. *)
Lemma handler_ok_enter (ctx : option code_id)
    (rets : finite_map (list flambda_kind)) (en : rebuild_env)
    (ce : finite_map (list node)) (conts : cont_env)
    (params : list variable) (md : handler_metadata) (body : expr) :
  handler_ok ctx rets en ce conts (Handler params md body) ->
  length params = length (hm_params_kinds md) ->
  expr_ok ctx rets (bind_variables en (with_kinds params (hm_params_kinds md)))
    ce conts body.
Proof.
  intros (Hl & Hb & Hs & Hcf & Hf & Hr) Hlen.
  split; [ exact Hl | split; [ exact Hb | split; [ exact Hs | ] ] ].
  split; [ exact Hcf | split; [ exact Hf | ] ].
  intros r K HK. destruct (Hr r K HK) as [H1 H2]. split; [ exact H1 | ].
  cbn [handler_ret_calls_ok] in H2. rewrite Hlen, Nat.eqb_refl in H2.
  exact H2.
Qed.

Lemma handler_entered (mask : list bool) (c c'' : cont_value)
    (vs vs' : list value) (statics statics' : static_env) (hp : heap)
    (rcd rcd' : read_record) :
  cont_rel mask c c'' ->
  statics_rel statics statics' ->
  map kind_of_value vs' = map kind_of_value vs ->
  (∀ (i : nat) (p : variable) (v v' : value),
     nth_error (cont_params c) i = Some (Node_variable p) ->
     nth_error vs i = Some v -> nth_error vs' i = Some v' ->
     is_var_used sol p (kind_of_value v) = true ->
     rel v (Node_variable p) v') ->
  match apply_cont_value c vs statics hp rcd with
  | Step s' =>
    flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc s' ->
    map kind_of_value vs = cont_kinds c
    ∧ ∃ t' : state,
        apply_cont_value c'' (filter_kept mask vs') statics' hp rcd' = Step t'
        ∧ related s' t'
  | Exit _ _ _ _ => False
  | Undefined => True
  end.
Proof.
  intros Hcrel Hst Hkinds Hargs.
  destruct Hcrel as [h en0 vars1 vars1' conts1 conts1' ctx0 ce0 rets0 Henv0
                       Hok0 Hedges0 Hconts0
                    | j h en0 vars1 vars1' conts1 conts1' handlers ctx0 ce0
                        rets0 Henv0 Hin0 Hall0];
    destruct h as [params metadata handler];
    cbn [apply_cont_value];
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hkb; try exact I;
    apply kinds_eqb_eq in Hkb;
    destruct (bind_params params vs vars1) as [hv | ] eqn:Hbind;
      try exact I;
    intros Hflow'; (split; [ exact Hkb | ]);
    assert (Hlen_h : length params = length (hm_params_kinds metadata))
      by (rewrite <- (bind_params_length _ _ _ _ Hbind), <- Hkb, length_map;
          reflexivity);
    destruct (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
      as [Hhv _];
    simpl_state;
    pose proof (params_not_dead params (hm_params_kinds metadata) vs vars1 hv
                  Hbind Hhv) as Hnodead;
    assert (Hargs' : ∀ (i : nat) (p : variable) (v v' : value),
              nth_error params i = Some p -> nth_error vs i = Some v ->
              nth_error vs' i = Some v' ->
              is_var_used sol p (kind_of_value v) = true ->
              rel v (Node_variable p) v')
      by (intros i p v v' Hp; apply Hargs; cbn [cont_params handler_params];
          rewrite nth_error_map, Hp; reflexivity);
    rewrite rebuild_handler_eq.
  - rewrite Hnodead. cbn [andb].
    destruct (enter_handler
                (handler_is_exn_handler (Handler params metadata handler))
                params metadata handler
                (rebuild (bind_variables en0
                            (with_kinds params (hm_params_kinds metadata)))
                   handler)
                en0 vars1 vars1' hv conts1' [] vs vs' statics' hp rcd' Henv0 Hkb
                Hkinds Hbind Hargs')
      as (hv' & Hstep & Hok').
    eexists. split; [ exact Hstep | ].
    split; [ exact Hflow' | ].
    eexists ctx0, _, ce0, rets0. simpl_state.
    split; [ exact Hedges0 | split; [ reflexivity | ] ].
    split; [ exact Hok' | ].
    split; [ | split; [ exact Hst | split; [ reflexivity | ] ] ];
      [ | split; [ exact (handler_ok_enter _ _ _ _ _ _ _ _ Hok0 Hlen_h)
                 | left; reflexivity ] ].
    rewrite rebuild_handler_eq, Hnodead in Hconts0. exact Hconts0.
  - cbn [andb].
    destruct (enter_handler false params metadata handler
                (rebuild (bind_variables (rec_env en0 handlers)
                            (with_kinds params (hm_params_kinds metadata)))
                   handler)
                (rec_env en0 handlers) vars1 vars1' hv conts1'
                (rebuilt_handlers (rec_env en0 handlers) handlers) vs vs'
                statics' hp rcd' Henv0 Hkb Hkinds Hbind Hargs')
      as (hv' & Hstep & Hok').
    eexists. split; [ exact Hstep | ].
    split; [ exact Hflow' | ].
    destruct (Hall0 j _ Hin0) as (Hok_h & Hedges_h & Hconts_h).
    eexists ctx0, _, (rec_conts ce0 handlers),
      (drop_rets (map fst handlers) rets0).
    simpl_state.
    split; [ exact Hedges_h | split; [ reflexivity | ] ].
    split; [ exact Hok' | ].
    split; [ | split; [ exact Hst | split; [ reflexivity | ] ] ];
      [ | split; [ exact (handler_ok_enter _ _ _ _ _ _ _ _ Hok_h Hlen_h)
                 | left; reflexivity ] ].
    rewrite rebuild_handler_eq in Hconts_h. cbn [andb] in Hconts_h.
    exact (rec_conts_related en0 handlers vars1 vars1' conts1 conts1' ctx0
             ce0 rets0 _ Henv0 Hall0 Hconts_h).
Qed.

Lemma filter_kept_true {A : Type} (mask : list bool) (l : list A) :
  forallb (fun keep => keep) mask = true -> filter_kept mask l = l.
Proof.
  revert l. induction mask as [ | b mask IH]; intros l H.
  { induction l as [ | a l IHl]; simpl; [ reflexivity | ].
    rewrite IHl; reflexivity. }
  simpl in H. apply andb_true_iff in H as [-> H].
  destruct l as [ | a l]; simpl; [ reflexivity | rewrite (IH l H) ];
    reflexivity.
Qed.

Lemma filter_kept_hd {A : Type} (mask : list bool) (a : A) (l : list A) :
  hd true mask = true ->
  filter_kept mask (a :: l) = a :: filter_kept (tl mask) l.
Proof.
  destruct mask as [ | b mask]; simpl; [ reflexivity | ].
  intros ->. reflexivity.
Qed.

Lemma apply_with_extra_args (c : cont_value) (extra vs : list value)
    (statics : static_env) (hp : heap) (rcd : read_record) :
  apply_cont_value (with_extra_args c extra) vs statics hp rcd
  = apply_cont_value c (vs ++ extra) statics hp rcd.
Proof.
  destruct extra as [ | x extra]; simpl; [ rewrite app_nil_r | ];
    reflexivity.
Qed.

Lemma names_from_length (n : nat) (p : positive) :
  length (names_from n p) = n.
Proof.
  revert p. induction n as [ | n IH]; intros p; simpl;
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma names_from_lt (n : nat) (p x : positive) :
  x ∈ names_from n (Pos.succ p) -> (p < x)%positive.
Proof.
  revert p. induction n as [ | n IH]; intros p H; simpl in H; [ destruct H | ].
  destruct H as [<- | H]; [ lia | specialize (IH (Pos.succ p) H); lia ].
Qed.

Lemma names_from_nodup (n : nat) (p : positive) : NoDup (names_from n p).
Proof.
  revert p. induction n as [ | n IH]; intros p; simpl; constructor.
  - intros H. apply names_from_lt in H. lia.
  - apply IH.
Qed.

Lemma bind_params_some (params : list variable) (vs : list value)
    (vars : env) :
  length params = length vs -> ∃ hv : env, bind_params params vs vars = Some hv.
Proof.
  revert vs vars. induction params as [ | x params IH];
    intros [ | v vs] vars H; simpl in H |- *; try discriminate H.
  - eexists; reflexivity.
  - apply (IH vs ((x, v) :: vars)). lia.
Qed.

Lemma bind_params_lookup_other (params : list variable) (vs : list value)
    (vars hv : env) (x : variable) :
  ~ x ∈ params -> bind_params params vs vars = Some hv ->
  lookup x hv = lookup x vars.
Proof.
  revert vs vars. induction params as [ | y params IH];
    intros [ | v vs] vars Hx H; simpl in H; try discriminate H.
  - injection H as <-. reflexivity.
  - rewrite (IH vs _ (fun Hin => Hx (or_intror Hin)) H). simpl.
    destruct (Pos.eqb_spec x y) as [-> | _];
      [ destruct (Hx (or_introl eq_refl)) | reflexivity ].
Qed.

Lemma eval_bound_params (params : list variable) (vs : list value)
    (vars hv : env) :
  NoDup params -> bind_params params vs vars = Some hv ->
  eval_simples hv (map Var params) = Some vs.
Proof.
  revert vs vars. induction params as [ | x params IH];
    intros [ | v vs] vars Hndp H; simpl in H; try discriminate H.
  - reflexivity.
  - inversion Hndp as [ | ? ? Hx Hnd']; subst.
    cbn [map eval_simples]. rewrite (IH vs _ Hnd' H).
    unfold eval_simple.
    rewrite (bind_params_lookup_other params vs _ hv x Hx H). simpl.
    rewrite Pos.eqb_refl. reflexivity.
Qed.


(* What the wrapper of a call that changes its calling convention gives
   the continuation of the call: for each result that the continuation
   keeps, what the callee returns, or poison where it returns nothing. *)
Definition poisoned (rd : list param_decision) (kinds : list flambda_kind)
    (vs : list value) : list value :=
  map
    (fun '((d, kind), v) => if is_kept d then v else Value_const (Poison kind))
    (combine (combine rd kinds) vs).

Lemma wrapper_args_eval (rd : list param_decision) :
  ∀ (kinds : list flambda_kind) (next : positive) (amask : list bool)
    (vs : list value) (rest wvars : env),
    length kinds = length rd -> length vs = length rd ->
    length amask = length rd ->
    bind_params (names_from (length (kept_kinds rd)) next)
      (filter_kept (map is_kept rd) vs) rest
    = Some wvars ->
    eval_simples wvars (wrapper_args next amask (combine rd kinds))
    = Some (filter_kept amask (poisoned rd kinds vs)).
Proof.
  induction rd as [ | d rd IH];
    intros kinds next amask vs rest wvars Lk Lv La Hbind.
  - destruct kinds; [ | discriminate Lk ].
    destruct amask; [ | discriminate La ].
    destruct vs; [ reflexivity | discriminate Lv ].
  - destruct kinds as [ | kind kinds]; [ discriminate Lk | ].
    destruct vs as [ | v vs]; [ discriminate Lv | ].
    destruct amask as [ | b amask]; [ discriminate La | ].
    injection Lk as Lk. injection Lv as Lv. injection La as La.
    destruct d as [k | ].
    + cbn [kept_kinds flat_map app length names_from map is_kept filter_kept
           bind_params] in Hbind.
      fold (kept_kinds rd) in Hbind.
      cbn [combine wrapper_args]. rewrite eval_simples_app.
      rewrite (IH kinds (Pos.succ next) amask vs ((next, v) :: rest) wvars
                 Lk Lv La Hbind).
      unfold poisoned. cbn [combine map is_kept].
      destruct b; cbn [eval_simples eval_simple filter_kept app];
        [ | reflexivity ].
      assert (Hnot : ~ next ∈ names_from (length (kept_kinds rd))
                                 (Pos.succ next)).
      { intros Hin. apply names_from_lt in Hin. lia. }
      rewrite (bind_params_lookup_other _ _ _ _ next Hnot Hbind).
      cbn [lookup]. rewrite Pos.eqb_refl. reflexivity.
    + cbn [kept_kinds flat_map app map is_kept filter_kept] in Hbind.
      fold (kept_kinds rd) in Hbind.
      cbn [combine wrapper_args]. rewrite eval_simples_app.
      rewrite (IH kinds next amask vs rest wvars Lk Lv La Hbind).
      unfold poisoned. cbn [combine map is_kept].
      destruct b; reflexivity.
Qed.

Lemma kept_kinds_filter (rd : list param_decision) :
  ∀ kinds : list flambda_kind,
    length rd = length kinds ->
    (∀ (i : nat) (kind : flambda_kind),
       nth_error rd i = Some (Keep kind) -> nth_error kinds i = Some kind) ->
    kept_kinds rd = filter_kept (map is_kept rd) kinds.
Proof.
  induction rd as [ | d rd IH]; intros [ | kind kinds] L Hk;
    try discriminate L; [ reflexivity | ].
  injection L as L.
  assert (Hk' : ∀ (i : nat) (kind' : flambda_kind),
            nth_error rd i = Some (Keep kind') ->
            nth_error kinds i = Some kind')
    by (intros i kind' H; exact (Hk (S i) kind' H)).
  destruct d as [k | ]; cbn [kept_kinds flat_map app map is_kept filter_kept];
    fold (kept_kinds rd); rewrite (IH kinds L Hk'); [ | reflexivity ].
  pose proof (Hk 0 k eq_refl) as H0. injection H0 as ->. reflexivity.
Qed.

Lemma map_kind_poisoned (rd : list param_decision) :
  ∀ (kinds : list flambda_kind) (vs : list value),
    length rd = length kinds ->
    map kind_of_value vs = kinds ->
    map kind_of_value (poisoned rd kinds vs) = kinds.
Proof.
  induction rd as [ | d rd IH]; intros [ | kind kinds] [ | v vs] L Hk;
    try discriminate L; try discriminate Hk; [ reflexivity | ].
  injection L as L. cbn [map] in Hk. injection Hk as Hv Hk.
  unfold poisoned in *. cbn [combine map].
  rewrite (IH kinds vs L Hk).
  destruct (is_kept d); [ rewrite Hv | ]; reflexivity.
Qed.

Lemma find_arm_in (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> (i, ac) ∈ arms.
Proof.
  induction arms as [ | [d ac'] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb_spec i d) as [<- | _].
  - intros [= <-]. left. reflexivity.
  - intros H. right. exact (IH H).
Qed.

Lemma find_arm_lookup (i : Z) (arms : list (Z * apply_cont)) :
  find_arm i arms = Keyed.lookup Z.eqb i arms.
Proof.
  induction arms as [ | [d ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i d); [ reflexivity | exact IH ].
Qed.

Lemma find_arm_compact_eq (i : Z) (arms : list (Z * apply_cont)) :
  find_arm i (Keyed.compact Z.eqb arms) = find_arm i arms.
Proof.
  rewrite !find_arm_lookup.
  exact (keyed_lookup_compact Z Z.eqb z_eqb_decides_equality _ i arms).
Qed.

(* The arm found among the arms that a map keeps or drops, keeping their
   discriminants. *)
Lemma find_arm_flat_map (f : Z * apply_cont -> list (Z * apply_cont))
    (i : Z) (arms : list (Z * apply_cont)) (ac ac' : apply_cont) :
  (∀ (d : Z) (a : apply_cont),
     f (d, a) = [] ∨ ∃ a' : apply_cont, f (d, a) = [(d, a')]) ->
  find_arm i arms = Some ac -> f (i, ac) = [(i, ac')] ->
  find_arm i (flat_map f arms) = Some ac'.
Proof.
  intros Hf. induction arms as [ | [d a] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb_spec i d) as [<- | Hne].
  - intros [= <-] Hfi. rewrite Hfi. simpl. rewrite Z.eqb_refl. reflexivity.
  - intros Harm Hfi.
    destruct (Hf d a) as [-> | (a' & ->)]; simpl; [ exact (IH Harm Hfi) | ].
    destruct (Z.eqb_spec i d); [ contradiction | exact (IH Harm Hfi) ].
Qed.

Definition rebuilt_arms (en : rebuild_env) (arms : list (Z * apply_cont))
    : list (Z * apply_cont) :=
  flat_map
    (fun '(d, ac) =>
       match rewrite_apply_cont sol en ac with
       | Some ac => [(d, ac)]
       | None => []
       end)
    (Keyed.compact Z.eqb arms).

Lemma rebuild_switch_eq (en : rebuild_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  rebuild en (Switch scrutinee arms)
  = if Keyed.is_empty (rebuilt_arms en arms) then Invalid
    else Switch (rewrite_simple sol en scrutinee) (rebuilt_arms en arms).
Proof. reflexivity. Qed.

Lemma switch_answered (scrutinee : simple) (arms : list (Z * apply_cont))
    (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Switch scrutinee arms) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Switch scrutinee arms) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  related_intro.
  unfold answered. rewrite step_switch in Hflow' |- *.
  destruct (eval_simple vars scrutinee) as [w | ] eqn:Hw; [ | exact I ].
  destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
  destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
  pose proof (proj1 (every_in _ _ _) He (i, ac) (find_arm_compact _ _ _ Harm))
    as Hac.
  destruct Hok as (Hlive & (bound & Hbound & Hexits) & Hscr & _ & Hkd
                   & Hnr).
  cbn [exits_bound scrutinees_used] in Hexits, Hscr.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs.
  2: { left. rewrite step_apply_cont, Hvs. reflexivity. }
  right. left.
  pose proof (rewrite_apply_cont_some en ac
                (args_have_sources vars _ vs Hvars (proj2 Hac) Hvs)) as Hrw.
  assert (Harm' : find_arm i (rebuilt_arms en arms)
                  = Some (rebuilt_apply_cont en ac)).
  { apply (find_arm_flat_map _ i _ ac).
    - intros d a. destruct (rewrite_apply_cont sol en a);
        [ right; eexists; reflexivity | left; reflexivity ].
    - rewrite find_arm_compact_eq. exact Harm.
    - cbv beta iota. rewrite Hrw. reflexivity. }
  rewrite rebuild_switch_eq in Ht.
  destruct (rebuilt_arms en arms) as [ | arm arms'] eqn:Harms;
    [ discriminate Harm' | ].
  rewrite <- Harms in Ht, Harm'. cbn [Keyed.is_empty] in Ht.
  rewrite Harms in Ht. cbn [Keyed.is_empty] in Ht. rewrite <- Harms in Ht.
  subst e'.
  assert (Hau_s : au (node_of_simple scrutinee)).
  { destruct ctx as [id | ];
      [ exact (any_usage_use _ _ _ Hc _ _ (au_hu _ _ _ Hc _ Hlive) Hscr)
      | exact (any_usage_input _ _ _ Hc _ Hscr) ]. }
  destruct (rewrite_simple_au en vars vars' scrutinee w Henv
              (or_intror Hau_s) Hw)
    as (w' & Hw' & Hel).
  assert (Hww : w' = w).
  { apply element_rel_sim in Hel.
    destruct w; simpl in Hi; try discriminate Hi; exact Hel. }
  subst w'.
  eexists. split.
  - rewrite step_switch, Hw', Hi, Harm'. reflexivity.
  - split; [ exact Hflow' | ].
    exists ctx, en, ce, rets. simpl_state. split; [ exact Hac | ]. split.
    + change (rebuild en (Apply_cont ac)) with (rebuild_apply_cont sol en ac).
      unfold rebuild_apply_cont. rewrite Hrw. reflexivity.
    + split; [ exact Henv | ].
      split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
      * apply (conts_related_sub _ _ _ _ _ _ _ Hconts). intros j Hj. simpl.
        apply in_flat_map. exists (i, rebuilt_apply_cont en ac).
        split; [ exact (find_arm_in _ _ _ Harm') | exact Hj ].
      * split; [ | left; apply (pending_not_let en vars vars' _ Hpend);
                   intros ? ? [ | ? ? ]; reflexivity ].
        split; [ exact Hlive | ].
        split; [ | split; [ exact I | split; [ reflexivity | ] ] ].
        -- exists bound. split; [ exact Hbound | ].
           cbn [exits_bound]. rewrite forallb_forall in Hexits.
           exact (Hexits (i, ac) (find_arm_in _ _ _ Harm)).
        -- split; [ exact (fresh_kinded_arm _ _ _ _ _ _ _ Hkd
                             (find_arm_in _ _ _ Harm)) | ].
           intros r K Hr. destruct (Hnr r K Hr) as [Hr1 Hr2].
           pose proof (find_arm_in _ _ _ Harm) as Hin.
           split.
           ++ reflexivity.
           ++ pose proof Hr2 as H.
              cbn [ret_calls_ok] in H |- *. rewrite forallb_forall in H.
              exact (H (i, ac) Hin).
Qed.

(* A step of the first run is answered by the second. *)
(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma call_witness_not_local :
  is_local (Field_call_witness Known_arity_code_pointer) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

Lemma fact_constructor (a b : node) (f : field) :
  (a, f, b) ∈ f_constructor F -> g_constructor G a f b.
Proof. rewrite HG. exact (fun H => H). Qed.

Lemma constructor_fact (a b : node) (f : field) :
  g_constructor G a f b -> (a, f, b) ∈ f_constructor F.
Proof. rewrite HG. exact (fun H => H). Qed.

Lemma fact_parameter (a b : node) (i : nat) :
  (a, i, b) ∈ f_parameter F -> g_parameter G a i b.
Proof. rewrite HG. exact (fun H => H). Qed.

Lemma parameter_fact (a b : node) (i : nat) :
  g_parameter G a i b -> (a, i, b) ∈ f_parameter F.
Proof. rewrite HG. exact (fun H => H). Qed.

Lemma accessor_fact (a b : node) (f : field) :
  g_accessor G a f b -> (a, f, b) ∈ f_accessor F.
Proof. rewrite HG. exact (fun H => H). Qed.

Lemma field_eqb_true (f f' : field) : field_eqb f f' = true <-> f = f'.
Proof. unfold field_eqb. destruct (field_eq_dec f f'); split; congruence. Qed.

Lemma node_eqb_true (n n' : node) : node_eqb n n' = true <-> n = n'.
Proof. unfold node_eqb. destruct (node_eq_dec n n'); split; congruence. Qed.

Lemma nth_error_seq_eq (start n i j : nat) :
  nth_error (seq start n) i = Some j -> j = start + i.
Proof.
  revert start i. induction n as [ | n IH]; intros start [ | i] H;
    simpl in H; try discriminate H.
  - injection H as <-. lia.
  - rewrite (IH (S start) i H). lia.
Qed.

Lemma nth_error_seq_lt (start n i : nat) :
  i < n -> nth_error (seq start n) i = Some (start + i).
Proof.
  revert start i. induction n as [ | n IH]; intros start [ | i] H; simpl;
    try lia.
  - f_equal. lia.
  - rewrite (IH (S start) i ltac:(lia)). f_equal. lia.
Qed.

Lemma nth_error_repeat_lt {A : Type} (x : A) (n i : nat) :
  i < n -> nth_error (repeat x n) i = Some x.
Proof.
  revert i. induction n as [ | n IH]; intros [ | i] H; simpl; try lia;
    [ reflexivity | apply IH; lia ].
Qed.

Lemma filter_kept_nil {A : Type} (l : list A) : filter_kept [] l = l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma filter_kept_all {A : Type} (n : nat) (l : list A) :
  filter_kept (repeat true n) l = l.
Proof.
  revert l. induction n as [ | n IH]; intros l; [ apply filter_kept_nil | ].
  destruct l as [ | a l]; simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

(* A read into a node that has a usage gives what is read a usage. *)
Lemma read_hu (a base to_ from : node) (f : field) :
  is_local f = false -> g_constructor G a f from -> g_accessor G to_ f base ->
  may_come_from sol a base -> hu to_ -> hu from.
Proof.
  intros Hl Hcons Hacc (_ & Hsrc & _ & Hau & Hus) Hto.
  destruct (s_any_source sol base) eqn:Eb.
  - destruct (s_any_usage sol a) eqn:Ea.
    + exact (au_hu _ _ _ Hc _
               (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hl)).
    + pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hto Hacc) as Hhu.
      assert (Eau : s_any_usage sol base = false).
      { destruct (s_any_usage sol base);
          [ discriminate (Hau eq_refl) | reflexivity ]. }
      pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
      destruct (Hus Hhu base Hbb) as [Hab | Hab]; [ | congruence ].
      exact (has_usage_alias _ _ _ Hc _ _ Hto
               (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Ea
                  Hacc)).
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    exact (has_usage_alias _ _ _ Hc _ _ Hto
             (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hacc Hsrc Eb Hcons)).
Qed.

(* The mask of the arguments of a direct call: all kept if what the
   callee may be, or its call witnesses, have any source; otherwise the
   arguments at the parameters that have a use of the witnesses that the
   callee may hold. *)
Lemma known_mask_cases (callee : node) (args : list simple) :
  (arguments_used_by_known_arity_call F sol callee args
   = repeat true (length args)
   ∧ (s_any_source sol callee = true
      ∨ ∃ x y : node,
          (x, Field_call_witness Known_arity_code_pointer, y) ∈ f_constructor F
          ∧ s_sources sol callee x = true ∧ s_any_source sol y = true))
  ∨ ∃ ws : node -> bool,
      arguments_used_by_known_arity_call F sol callee args
      = map (cofield_has_use F sol ws) (seq 0 (length args))
      ∧ s_any_source sol callee = false
      ∧ (∀ x y : node,
           (x, Field_call_witness Known_arity_code_pointer, y)
           ∈ f_constructor F ->
           s_sources sol callee x = true -> s_has_source sol x = true ->
           s_any_source sol y = false)
      ∧ ∀ z : node,
          ws z = true <->
          ∃ x y : node,
            (x, Field_call_witness Known_arity_code_pointer, y)
            ∈ f_constructor F
            ∧ s_sources sol callee x = true ∧ s_has_source sol x = true
            ∧ s_sources sol y z = true ∧ s_has_source sol z = true.
Proof.
  unfold arguments_used_by_known_arity_call, arguments_used_by_call,
    get_direct_sources.
  destruct (s_any_source sol callee) eqn:Ec.
  { left. split; [ simpl; apply app_nil_r | left; reflexivity ]. }
  unfold get_field_sources.
  destruct (existsb _ (f_constructor F)) eqn:Hex.
  { left. split; [ simpl; apply app_nil_r | right ].
    apply existsb_exists in Hex as ([[x f] y] & Hin & H).
    apply andb_true_iff in H as [H Hy]. apply andb_true_iff in H as [Hx Hf].
    apply andb_true_iff in Hx as [Hx _]. apply field_eqb_true in Hf. subst f.
    exists x, y. auto. }
  right. eexists. split; [ simpl; rewrite ?app_nil_r; reflexivity | ].
  split; [ reflexivity | split ].
  - intros x y Hin Hs Hhs. destruct (s_any_source sol y) eqn:Ey;
      [ | reflexivity ].
    apply not_true_iff_false in Hex. exfalso. apply Hex.
    apply existsb_exists.
    exists (x, Field_call_witness Known_arity_code_pointer, y).
    split; [ exact Hin | ].
    rewrite Hs, Hhs, Ey, (proj2 (field_eqb_true _ _) eq_refl). reflexivity.
  - intros z. split.
    + intros Hz. apply existsb_exists in Hz as ([[x f] y] & Hin & H).
      apply andb_true_iff in H as [H Hhz]. apply andb_true_iff in H as [H Hyz].
      apply andb_true_iff in H as [Hx Hf].
      apply andb_true_iff in Hx as [Hsx Hhx]. apply field_eqb_true in Hf.
      subst f. exists x, y. auto.
    + intros (x & y & Hin & Hsx & Hhx & Hyz & Hhz). apply existsb_exists.
      exists (x, Field_call_witness Known_arity_code_pointer, y).
      split; [ exact Hin | ].
      rewrite Hsx, Hhx, Hyz, Hhz, (proj2 (field_eqb_true _ _) eq_refl).
      reflexivity.
Qed.

(* An argument that the mask of a direct call keeps has a use. *)
Lemma known_mask_kept (callee witness apply_node : node) (args : list simple)
    (i : nat) (s : simple) :
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    callee ->
  s_alias sol witness apply_node = true ->
  g_argument G (node_of_simple s) i witness ->
  nth_error (arguments_used_by_known_arity_call F sol callee args) i
  = Some true ->
  hu (node_of_simple s).
Proof.
  intros Happly Hal Harg Hmask.
  destruct (s_any_source sol witness) eqn:Ew.
  { exact (au_hu _ _ _ Hc _ (any_usage_argument _ _ _ Hc _ _ _ Ew Harg)). }
  assert (Ea : s_any_source sol apply_node = false).
  { destruct (s_any_source sol apply_node) eqn:Ea; [ | reflexivity ].
    rewrite (any_source_alias _ _ _ Hc _ _ Hal Ea) in Ew. discriminate Ew. }
  destruct (known_mask_cases callee args)
    as [[_ [Ec | (x & y & Hxy & Hsx & Ey)]] | (ws & Hws & Ec & Hy & Hz)].
  - rewrite (any_source_accessor _ _ _ Hc _ _ _ Ec Happly
               call_witness_not_local) in Ea.
    discriminate Ea.
  - destruct (s_any_source sol callee) eqn:Ec.
    + rewrite (any_source_accessor _ _ _ Hc _ _ _ Ec Happly
                 call_witness_not_local) in Ea.
      discriminate Ea.
    + pose proof (alias_accessor_source _ _ _ Hc _ _ _ _ _ Happly Hsx Ec
                    (fact_constructor _ _ _ Hxy)) as Hay.
      rewrite (any_source_alias _ _ _ Hc _ _ Hay Ey) in Ea. discriminate Ea.
  - rewrite Hws, nth_error_map in Hmask.
    destruct (nth_error (seq 0 (length args)) i) as [j | ] eqn:Hj;
      [ | discriminate Hmask ].
    rewrite (nth_error_seq_eq _ _ _ _ Hj) in Hmask. simpl in Hmask.
    injection Hmask as Hmask.
    unfold cofield_has_use in Hmask.
    apply existsb_exists in Hmask as ([[z j'] p] & Hzp & Hb).
    apply andb_true_iff in Hb as [Hb Hp].
    apply andb_true_iff in Hb as [Hwz Hji].
    apply Nat.eqb_eq in Hji. subst j'.
    destruct (proj1 (Hz z) Hwz) as (x & y & Hxy & Hsx & Hhx & Hsyz & _).
    pose proof (alias_accessor_source _ _ _ Hc _ _ _ _ _ Happly Hsx Ec
                  (fact_constructor _ _ _ Hxy)) as Hay.
    pose proof (Hy x y Hxy Hsx Hhx) as Ey.
    pose proof (sources_alias _ _ _ Hc _ _ _ Hsyz Ey Hay Ea) as Haz.
    pose proof (sources_alias _ _ _ Hc _ _ _ Haz Ea Hal Ew) as Hwz'.
    pose proof (alias_argument_source _ _ _ Hc _ _ _ _ _ Harg Hwz' Ew
                  (fact_parameter _ _ _ Hzp)) as Hpa.
    exact (has_usage_alias _ _ _ Hc _ _ Hp Hpa).
Qed.

(* A parameter that has a use, of the witness of code that the callee of
   a direct call may be a closure of, has its argument kept. *)
Lemma known_mask_used (callee b w p : node) (args : list simple) (i : nat) :
  may_come_from sol b callee -> may_come_from sol b b ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  may_come_from sol w w ->
  g_parameter G w i p -> hu p -> i < length args ->
  nth_error (arguments_used_by_known_arity_call F sol callee args) i
  = Some true.
Proof.
  intros Hb Hbb Hcw Hww Hpar Hp Hi.
  destruct (known_mask_cases callee args)
    as [[-> _] | (ws & -> & Ec & Hy & Hz)];
    [ exact (nth_error_repeat_lt true _ _ Hi) | ].
  rewrite nth_error_map, (nth_error_seq_lt 0 _ _ Hi). simpl. f_equal.
  destruct Hb as (_ & [Hcs | Hcs] & _); [ congruence | ].
  destruct Hbb as (Hhb & _).
  pose proof (Hy b w (constructor_fact _ _ _ Hcw) Hcs Hhb) as Ew.
  destruct Hww as (Hhw & [Hws | Hws] & _); [ congruence | ].
  unfold cofield_has_use. apply existsb_exists.
  exists (w, i, p). split; [ exact (parameter_fact _ _ _ Hpar) | ].
  rewrite Nat.eqb_refl, Hp, !andb_true_r.
  apply (proj2 (Hz w)). exists b, w.
  split; [ exact (constructor_fact _ _ _ Hcw) | auto ].
Qed.

(* The witness of the code of what the callee of a direct call may be is
   among those the call is found to call. *)
Lemma directly_called_in (closure apply_node b w : node) (id : code_id) :
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    closure ->
  may_come_from sol b closure ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
  may_come_from sol w w ->
  ∀ ids : list code_id,
    code_id_actually_directly_called F sol closure = Some ids -> id ∈ ids.
Proof.
  intros Happly Hb Hcw Hcode Hww ids. unfold code_id_actually_directly_called.
  set (widgets := filter _ (f_accessor F)).
  assert (Hin : (apply_node, Field_call_witness Known_arity_code_pointer,
                 closure) ∈ widgets).
  { apply filter_In. split; [ exact (accessor_fact _ _ _ Happly) | ].
    rewrite (proj2 (node_eqb_true _ _) eq_refl),
      (proj2 (field_eqb_true _ _) eq_refl).
    reflexivity. }
  destruct (existsb _ widgets) eqn:Has; [ discriminate | ].
  intros [= <-].
  assert (Ea : s_any_source sol apply_node = false).
  { apply not_true_iff_false. intros Ea. apply not_true_iff_false in Has.
    apply Has. apply existsb_exists. eexists. split; [ exact Hin | exact Ea ].
  }
  assert (Ec : s_any_source sol closure = false).
  { apply not_true_iff_false. intros Ec.
    rewrite (any_source_accessor _ _ _ Hc _ _ _ Ec Happly
               call_witness_not_local) in Ea.
    discriminate Ea. }
  destruct Hb as (_ & [Hcs | Hcs] & _); [ congruence | ].
  pose proof (alias_accessor_source _ _ _ Hc _ _ _ _ _ Happly Hcs Ec Hcw)
    as Haw.
  assert (Ew : s_any_source sol w = false).
  { apply not_true_iff_false. intros Ew.
    rewrite (any_source_alias _ _ _ Hc _ _ Haw Ew) in Ea. discriminate Ea. }
  destruct Hww as (Hhw & [Hws | Hws] & _); [ congruence | ].
  pose proof (sources_alias _ _ _ Hc _ _ _ Hws Ew Haw Ea) as Haw'.
  apply in_flat_map. eexists. split; [ exact Hin | ].
  apply in_flat_map.
  exists (w, Field_code_id_of_call_witness, Node_code_id id).
  split; [ exact (constructor_fact _ _ _ Hcode) | ].
  cbv beta iota. rewrite (proj2 (field_eqb_true _ _) eq_refl), Haw', Hhw.
  left. reflexivity.
Qed.

Lemma singleton_in (ids : list code_id) (id id' : code_id) :
  Keyed.get_singleton Pos.eqb ids = Some id' -> id ∈ ids -> id' = id.
Proof.
  unfold Keyed.get_singleton. intros Hs Hin.
  destruct (Keyed.dedup Pos.eqb ids) as [ | k [ | k' l]] eqn:Hd;
    try discriminate Hs.
  injection Hs as <-.
  pose proof (proj2 (keyed_among_in _ Pos.eqb Pos.eqb_eq id ids) Hin) as Ha.
  rewrite <- (keyed_among_dedup _ Pos.eqb Pos.eqb_eq id ids), Hd in Ha.
  apply (keyed_among_in _ Pos.eqb Pos.eqb_eq) in Ha.
  destruct Ha as [-> | []]. reflexivity.
Qed.

(* The call kind of the rebuilt call: the direct call of the code that
   the callee is a closure of, or an indirect call. *)
Lemma rewrite_call_kind_cases (ap : apply_expr) (id : code_id)
    (b w apply_node : node) :
  ap_call_kind ap = Direct id ->
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    (node_of_simple (ap_callee ap)) ->
  may_come_from sol b (node_of_simple (ap_callee ap)) ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
  may_come_from sol w w ->
  rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap = Indirect
  ∨ rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap
    = Direct id.
Proof.
  intros Hk Happly Hb Hcw Hcode Hww.
  pose proof (directly_called_in _ _ _ _ id Happly Hb Hcw Hcode Hww) as Hin.
  unfold rewrite_call_kind. rewrite Hk. cbv zeta.
  assert (Hunknown :
    (if preserve_direct_calls || negb (among id unit_code_ids)
     then Direct id else Indirect) = Indirect
    ∨ (if preserve_direct_calls || negb (among id unit_code_ids)
       then Direct id else Indirect) = Direct id)
    by (destruct (_ || _); auto).
  revert Hin.
  destruct (ap_callee ap) as [x | sym | c]; intros Hin; [ | | exact Hunknown ];
    destruct (code_id_actually_directly_called F sol _) as [ids | ];
    try exact Hunknown;
    specialize (Hin ids eq_refl);
    (destruct (Keyed.get_singleton Pos.eqb ids) as [id' | ] eqn:Hs;
     [ right; f_equal; exact (singleton_in ids id id' Hs Hin)
     | destruct (Keyed.is_empty ids); auto ]).
Qed.

(* Whether code changes its calling convention does not depend on its
   metadata. *)
Lemma code_change_metadata (id : code_id) (md md' : code_metadata) :
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    md
  = Not_changing_calling_convention ->
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    md'
  = Not_changing_calling_convention.
Proof.
  unfold code_calling_convention_change.
  destruct (cannot_change_calling_convention _ _ id); [ reflexivity | ].
  destruct (lookup id code_deps); [ discriminate | reflexivity ].
Qed.

(* A direct call known to call code calls the code of its callee. *)
Lemma code_id_actually_called_eq (ap : apply_expr) (id id' : code_id)
    (b w apply_node : node) :
  ap_call_kind ap = Direct id ->
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    (node_of_simple (ap_callee ap)) ->
  may_come_from sol b (node_of_simple (ap_callee ap)) ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
  may_come_from sol w w ->
  code_id_actually_called F sol ap = Some id' -> id' = id.
Proof.
  intros Hk Happly Hb Hcw Hcode Hww.
  pose proof (directly_called_in _ _ _ _ id Happly Hb Hcw Hcode Hww) as Hin.
  revert Hin. unfold code_id_actually_called. rewrite Hk.
  destruct (ap_callee ap) as [x | sym | c0]; intros Hin;
    [ | | intros H; discriminate H ];
    destruct (code_id_actually_directly_called F sol _) as [ids | ];
    try (intros H; discriminate H);
    specialize (Hin ids eq_refl); intros Hs;
    exact (singleton_in ids id id' Hs Hin).
Qed.

(* The rebuild of a direct call of code that keeps its calling convention
   is that of FlambdaReaperRebuild.v. *)
Lemma rebuild_apply_unchanged_code (ap : apply_expr) (id : code_id)
    (md : code_metadata) (en : rebuild_env) (b w apply_node : node) :
  ap_call_kind ap = Direct id ->
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    (node_of_simple (ap_callee ap)) ->
  may_come_from sol b (node_of_simple (ap_callee ap)) ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
  may_come_from sol w w ->
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    md
  = Not_changing_calling_convention ->
  rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids code_metadatas first_fresh en ap
  = rebuild_apply F sol preserve_direct_calls unit_code_ids first_fresh en ap.
Proof.
  intros Hk Happly Hb Hcw Hcode Hww Hun.
  unfold rebuild_apply_cc, apply_calling_convention_change.
  destruct (code_id_actually_called F sol ap) as [id' | ] eqn:Hcalled;
    [ | reflexivity ].
  rewrite (code_id_actually_called_eq ap id id' b w apply_node Hk Happly Hb
             Hcw Hcode Hww Hcalled).
  unfold get_calling_convention_change.
  destruct (lookup id code_metadatas) as [md' | ]; [ | reflexivity ].
  rewrite (code_change_metadata id md md' Hun). reflexivity.
Qed.

Lemma apply_cc_unchanged_code (ap : apply_expr) (id : code_id)
    (md : code_metadata) (b w apply_node : node) :
  ap_call_kind ap = Direct id ->
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    (node_of_simple (ap_callee ap)) ->
  may_come_from sol b (node_of_simple (ap_callee ap)) ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
  may_come_from sol w w ->
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    md
  = Not_changing_calling_convention ->
  apply_calling_convention_change F sol cannot_change code_deps
    unit_code_ids code_metadatas ap
  = Not_changing_calling_convention.
Proof.
  intros Hk Happly Hb Hcw Hcode Hww Hun.
  unfold apply_calling_convention_change.
  destruct (code_id_actually_called F sol ap) as [id' | ] eqn:Hcalled;
    [ | reflexivity ].
  rewrite (code_id_actually_called_eq ap id id' b w apply_node Hk Happly Hb
             Hcw Hcode Hww Hcalled).
  unfold get_calling_convention_change.
  destruct (lookup id code_metadatas) as [md' | ]; [ | reflexivity ].
  exact (code_change_metadata id md md' Hun).
Qed.

Lemma singleton_of_all (ids : list code_id) (id : code_id) :
  id ∈ ids -> (∀ id' : code_id, id' ∈ ids -> id' = id) ->
  Keyed.get_singleton Pos.eqb ids = Some id.
Proof.
  intros Hin Hall. unfold Keyed.get_singleton.
  destruct ids as [ | x rest]; [ destruct Hin | ].
  cbn [Keyed.dedup]. rewrite (Hall x (or_introl eq_refl)).
  assert (Hnil : ∀ l : list code_id,
            (∀ y : code_id, y ∈ l -> y = id) ->
            Keyed.without Pos.eqb [id] l = []).
  { induction l as [ | y l IH]; intros Hl; [ reflexivity | ].
    unfold Keyed.without in *. cbn [filter].
    rewrite (Hl y (or_introl eq_refl)). unfold Keyed.among. cbn [existsb].
    rewrite Pos.eqb_refl. cbn [orb negb].
    apply IH. intros z Hz. apply Hl. right. exact Hz. }
  rewrite Hnil; [ reflexivity | ].
  intros y Hy. apply Hall. right.
  apply (keyed_among_in _ Pos.eqb Pos.eqb_eq).
  rewrite <- (keyed_among_dedup _ Pos.eqb Pos.eqb_eq).
  apply (keyed_among_in _ Pos.eqb Pos.eqb_eq). exact Hy.
Qed.

(* A direct call of code that changes its calling convention is known to
   call it ([code_id_actually_called]): the witness of known arity of the
   code is used, by the call, and so cannot have another source, nor any,
   where the call reads it, nor where another call reads it off the
   callee ([Hwidgets]). *)
Lemma direct_call_recognized (ap : apply_expr) (id : code_id)
    (b w apply_node witness called_node : node) :
  ap_call_kind ap = Direct id ->
  (∀ c0 : const, ap_callee ap <> Const c0) ->
  g_accessor G apply_node (Field_call_witness Known_arity_code_pointer)
    (node_of_simple (ap_callee ap)) ->
  s_alias sol witness apply_node = true ->
  g_accessor G called_node Field_code_id_of_call_witness witness ->
  au called_node ->
  may_come_from sol b (node_of_simple (ap_callee ap)) ->
  g_constructor G b (Field_call_witness Known_arity_code_pointer) w ->
  g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
  may_come_from sol w w ->
  cannot_change (Node_code_id id) = false ->
  code_id_actually_called F sol ap = Some id.
Proof.
  intros Hk Hnc Happly Hal Hcalled Hau_called Hb Hcw Hcode Hww Hcc_id.
  assert (Hccw : hu w -> cannot_change_witness w = false).
  { intros Hhw. destruct (cannot_change_witness w) eqn:E; [ | reflexivity ].
    rewrite (cc_code_id_of_witness _ _ _ _ Hcc_closed w _ Hcode Hhw E)
      in Hcc_id.
    discriminate Hcc_id. }
  pose proof (au_hu _ _ _ Hc _ Hau_called) as Hhu_called.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_called Hcalled)
    as Hhu_wit.
  pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_wit Hal) as Hhu_a.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_a Happly) as Hhu_n.
  assert (Hau_w : s_any_usage sol w = false).
  { destruct (s_any_usage sol w) eqn:E; [ | reflexivity ].
    pose proof (Hccw (au_hu _ _ _ Hc _ E)) as H.
    rewrite (cc_witness_any_usage _ _ _ _ Hcc_closed w E) in H.
    discriminate H. }
  assert (Hau_b : s_any_usage sol b = false).
  { destruct (s_any_usage sol b) eqn:E; [ | reflexivity ].
    rewrite (any_usage_constructor _ _ _ Hc _ _ _ E Hcw
               call_witness_not_local) in Hau_w.
    discriminate Hau_w. }
  assert (Hau_n : s_any_usage sol (node_of_simple (ap_callee ap)) = false).
  { destruct (s_any_usage sol (node_of_simple (ap_callee ap))) eqn:E;
      [ | reflexivity ].
    destruct Hb as (_ & _ & _ & Hb4 & _). rewrite (Hb4 E) in Hau_b.
    discriminate Hau_b. }
  pose proof (usages_accessor _ _ _ Hc _ _ _ Happly Hau_n) as Hus_n.
  assert (Hus_bn : s_usages sol b (node_of_simple (ap_callee ap)) = true).
  { destruct Hb as (_ & _ & _ & _ & Hb5).
    destruct (Hb5 Hhu_n _ Hus_n) as [H | H]; [ exact H | congruence ]. }
  pose proof (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcw Hus_bn Hau_b
                Happly) as Hal_aw.
  pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_a Hal_aw) as Hhu_w.
  pose proof (Hccw Hhu_w) as Hccw_w.
  assert (Hau_a : s_any_usage sol apply_node = false).
  { destruct (s_any_usage sol apply_node) eqn:E; [ | reflexivity ].
    rewrite (any_usage_alias _ _ _ Hc _ _ E Hal_aw) in Hau_w.
    discriminate Hau_w. }
  assert (Hau_wit : s_any_usage sol witness = false).
  { destruct (s_any_usage sol witness) eqn:E; [ | reflexivity ].
    rewrite (any_usage_alias _ _ _ Hc _ _ E Hal) in Hau_a.
    discriminate Hau_a. }
  pose proof (usages_accessor _ _ _ Hc _ _ _ Hcalled Hau_wit) as Hus_wit.
  pose proof (usages_alias _ _ _ Hc _ _ _ Hus_wit Hau_wit Hal Hau_a)
    as Hus_a.
  pose proof (usages_alias _ _ _ Hc _ _ _ Hus_a Hau_a Hal_aw Hau_w)
    as Hus_w.
  assert (Has_wit : s_any_source sol witness = false).
  { destruct (s_any_source sol witness) eqn:E; [ | reflexivity ].
    rewrite (cc_witness_any_source_usage _ _ _ _ Hcc_closed w witness
               called_node Hus_w E Hcalled Hhu_called) in Hccw_w.
    discriminate Hccw_w. }
  assert (Has_a : s_any_source sol apply_node = false).
  { destruct (s_any_source sol apply_node) eqn:E; [ | reflexivity ].
    rewrite (any_source_alias _ _ _ Hc _ _ Hal E) in Has_wit.
    discriminate Has_wit. }
  (* The witnesses read off the callee. *)
  assert (Hwid :
    ∀ a' : node,
      (a', Field_call_witness Known_arity_code_pointer,
       node_of_simple (ap_callee ap)) ∈ f_accessor F ->
      s_any_source sol a' = false
      ∧ ∀ (cw : node) (id' : code_id),
          s_sources sol a' cw = true -> s_has_source sol cw = true ->
          g_constructor G cw Field_code_id_of_call_witness (Node_code_id id')
          -> id' = id).
  { intros a' Ha'.
    assert (Ga' : g_accessor G a' (Field_call_witness Known_arity_code_pointer)
                    (node_of_simple (ap_callee ap)))
      by (rewrite HG; exact Ha').
    split.
    - destruct (s_any_source sol a') eqn:E; [ | reflexivity ].
      destruct (Hwidgets id apply_node a' _ w Hcc_id Happly Ga')
        as [Has' _].
      rewrite (Has' E) in Has_a. discriminate Has_a.
    - intros cw id' Hsrc Hhs Hcw'.
      destruct (Hwidgets id apply_node a' _ cw Hcc_id Happly Ga')
        as [_ Hsrc'].
      pose proof (sources_alias _ _ _ Hc _ _ _ (Hsrc' Hsrc) Has_a Hal
                    Has_wit) as Hsrc_wit.
      destruct (node_eq_dec cw w) as [-> | Hne].
      + exact (Hcode_id_of_witness id id' w Hcc_id Hcode Hcw').
      + exfalso.
        rewrite (cc_witness_other_source _ _ _ _ Hcc_closed w witness cw
                   called_node Hus_w Hsrc_wit Hhs Hne Hcalled Hhu_called)
          in Hccw_w.
        discriminate Hccw_w. }
  pose proof (directly_called_in _ _ _ _ id Happly Hb Hcw Hcode Hww) as Hin.
  assert (Hall : ∀ ids : list code_id,
            code_id_actually_directly_called F sol
              (node_of_simple (ap_callee ap)) = Some ids ->
            ∀ id' : code_id, id' ∈ ids -> id' = id).
  { intros ids. unfold code_id_actually_directly_called.
    destruct (existsb _ _); [ discriminate | ].
    intros [= <-] id' Hid'.
    apply in_flat_map in Hid' as ([[a' f] base] & Hx & Hid').
    apply filter_In in Hx as [Hx Hf]. apply andb_true_iff in Hf as [Hb' Hf'].
    apply node_eqb_true in Hb'. apply field_eqb_true in Hf'.
    subst base f.
    apply in_flat_map in Hid' as ([[cw f] cid] & Hcx & Hid').
    destruct cid as [ | | id'' | | ]; try destruct Hid'.
    destruct (field_eqb f Field_code_id_of_call_witness
              && s_sources sol a' cw && s_has_source sol cw) eqn:Ec;
      [ | destruct Hid' ].
    destruct Hid' as [<- | []].
    apply andb_true_iff in Ec as [Ec Hhs].
    apply andb_true_iff in Ec as [Ef Hsrc].
    apply field_eqb_true in Ef. subst f.
    exact (proj2 (Hwid a' Hx) cw id'' Hsrc Hhs (fact_constructor _ _ _ Hcx)).
  }
  assert (Hsome : ∃ ids : list code_id,
            code_id_actually_directly_called F sol
              (node_of_simple (ap_callee ap)) = Some ids).
  { unfold code_id_actually_directly_called.
    match goal with
    | |- context [if existsb ?f ?l then _ else _] =>
      replace (existsb f l) with false; [ eexists; reflexivity | ]
    end.
    symmetry. apply not_true_iff_false. intros Hex.
    apply existsb_exists in Hex as ([[a' f] base] & Hx & Has').
    apply filter_In in Hx as [Hx Hf]. apply andb_true_iff in Hf as [Hb' Hf'].
    apply node_eqb_true in Hb'. apply field_eqb_true in Hf'.
    subst base f.
    rewrite (proj1 (Hwid a' Hx)) in Has'. discriminate Has'. }
  destruct Hsome as (ids & Hids).
  revert Hids Hin Hall. unfold code_id_actually_called. rewrite Hk.
  destruct (ap_callee ap) as [x | sym | c0];
    [ | | intros; exfalso; exact (Hnc c0 eq_refl) ];
    intros Hids Hin Hall; cbv iota beta; rewrite Hids;
    exact (singleton_of_all ids id (Hin ids Hids) (Hall ids Hids)).
Qed.

(* A closure of the first run, related at a node, is related at the node
   of its allocation, which stands for its code. *)
Lemma rel_closure_alloc (statics : static_env) (v v' : value) (n : node)
    (id : code_id) (slots : list (value_slot * value)) :
  statics_flow_ statics -> rel v n v' ->
  closure_of statics v = Some (id, slots) ->
  ∃ b : node, may_come_from sol b n ∧ rel v b v' ∧ called b id.
Proof.
  intros Hst Hv Hcl.
  destruct v as [ | code slots0 | slot decls slots0 | s | | | | | ];
    simpl in Hcl; try discriminate Hcl.
  - destruct Hv.
  - destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hcl ].
    injection Hcl as -> _.
    destruct Hv as (sites & vs' & Hn & Hv' & Hex & Hall).
    exists (sites slot). split; [ exact Hn | split ].
    + exists sites, vs'. split; [ exact (proj1 (Hall slot id Hl)) | ].
      exact (conj Hv' (conj Hex Hall)).
    + exact (proj1 (proj2 (Hall slot id Hl))).
  - destruct (lookup s (static_values statics))
      as [[ | slot syms decls slots0 | | | | | ] | ] eqn:Hls;
      try discriminate Hcl.
    + destruct (Hst s _ Hls) as [_ []].
    + destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hcl ].
      injection Hcl as -> _.
      destruct (Hst s _ Hls) as (Hself & Hcalled & _).
      destruct Hv as (-> & Hn & _).
      exists (Node_symbol s). split; [ exact Hn | split ].
      * split; [ reflexivity | split; [ exact Hself | auto ] ].
      * exact (Hcalled id Hl).
Qed.

(* The value of the second run is a closure of the same code. *)
Lemma rel_closure_of (statics statics' : static_env) (v v' : value)
    (n : node) (id : code_id) (slots : list (value_slot * value)) :
  statics_rel statics statics' -> rel v n v' -> hu n ->
  closure_of statics v = Some (id, slots) ->
  ∃ slots' : list (value_slot * value),
    closure_of statics' v' = Some (id, slots').
Proof.
  intros (_ & _ & Honly & _) Hv Hn Hcl.
  destruct v as [ | code slots0 | slot decls slots0 | s | | | | | ];
    simpl in Hcl; try discriminate Hcl.
  - destruct Hv.
  - destruct Hv as (sites & vs' & _ & -> & _).
    simpl. destruct (lookup slot decls) as [id0 | ]; [ | discriminate Hcl ].
    injection Hcl as -> _. eexists. reflexivity.
  - destruct Hv as (-> & _ & Hhu). specialize (Hhu Hn).
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hls;
      [ | discriminate Hcl ].
    destruct (Honly s allocated Hls)
      as [(tag & fields & ->)
         | [(slot & syms & decls & vs & -> & Hs' & Hused) | (Ho & _)]];
      [ discriminate Hcl | |
        destruct allocated; try discriminate Hcl; destruct Ho ].
    simpl. destruct Hs' as [Hnone | (vs' & Hs' & _)];
      [ destruct (Hused Hhu Hnone) | ].
    rewrite Hs'. destruct (lookup slot decls); [ | discriminate Hcl ].
    injection Hcl as -> _. eexists. reflexivity.
Qed.

(* A simple that the first run evaluates has a source. *)
Lemma simple_not_dead (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_flows_ vars -> simple_noted noted s -> eval_simple vars s = Some v ->
  simple_is_dead sol en s = false.
Proof.
  intros Hvars Hnoted Hs.
  pose proof (flows_hs is_local G sol keep called no_loc Hc _ _
                (simple_flows is_local G sol noted cds keep no_loc Hc Hconsts
                    Hsyms _
                   _ _ Hvars Hnoted Hs)) as Hhs.
  destruct s as [x | sym | c]; simpl in Hhs |- *; [ | rewrite Hhs; reflexivity
    | reflexivity ].
  unfold is_dead_var. destruct (kind_of_variable en x);
    try rewrite Hhs; reflexivity.
Qed.

(* The arguments of a call at a mask, in the second run: those kept, as
   they are, and poison for the others, of the same kinds. *)
Lemma mask_arguments_eval (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (mask : list bool) (args : list simple) (vs : list value),
    simples_noted noted args ->
    eval_simples vars args = Some vs ->
    length mask = length args ->
    (∀ (i : nat) (s : simple),
       nth_error args i = Some s -> nth_error mask i = Some true ->
       hu (node_of_simple s)) ->
    ∃ vs' : list value,
      eval_simples vars' (mask_arguments en mask args) = Some vs'
      ∧ map kind_of_value vs' = map kind_of_value vs
      ∧ ∀ (i : nat) (s : simple) (v v' : value),
          nth_error args i = Some s -> nth_error vs i = Some v ->
          nth_error vs' i = Some v' -> nth_error mask i = Some true ->
          rel v (node_of_simple s) v'.
Proof.
  intros Henv. induction mask as [ | m mask IH];
    intros [ | s args] vs Hnoted Hvs Hlen Hkept; simpl in Hlen;
    try discriminate Hlen.
  - simpl in Hvs. injection Hvs as <-. exists [].
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros [ | i]; discriminate.
  - simpl in Hvs.
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-.
    destruct (IH args ws (fun s' Hin => Hnoted s' (or_intror Hin)) Hws
                ltac:(lia) (fun i => Hkept (S i)))
      as (ws' & Hws' & Hks & Hrels).
    destruct (rewrite_simple_eval en vars vars' s v Henv
                (Hnoted s (or_introl eq_refl)) Hs)
      as (v1 & Hs1 & Hk1 & Hkept1).
    assert (Hp : ∃ v2 : value,
              eval_simple vars' (poison_argument en s) = Some v2
              ∧ kind_of_value v2 = kind_of_value v).
    { destruct s as [x | sym | c]; simpl in Hs |- *.
      - eexists. split; [ reflexivity | ]. simpl.
        exact (env_ok_kind en vars vars' x v Henv Hs).
      - injection Hs as <-. eexists. split; reflexivity.
      - injection Hs as <-. eexists. split; reflexivity. }
    destruct m.
    + pose proof (Hkept 0 s eq_refl eq_refl) as Hhu.
      destruct (Hkept1 (kept_of_hu en s Hhu)) as [Hrw Hrel].
      rewrite Hrw in Hs1.
      exists (v1 :: ws'). cbn [mask_arguments eval_simples].
      rewrite Hs1, Hws'.
      split; [ reflexivity | ].
      split; [ simpl; rewrite Hk1, Hks; reflexivity | ].
      intros [ | i] s'' w w' Hi Hw Hw'; simpl in Hi, Hw, Hw'.
      * injection Hi as <-. injection Hw as <-. injection Hw' as <-.
        intros _. exact Hrel.
      * exact (Hrels i s'' w w' Hi Hw Hw').
    + destruct Hp as (v2 & Hs2 & Hk2).
      exists (v2 :: ws'). cbn [mask_arguments eval_simples].
      rewrite Hs2, Hws'.
      split; [ reflexivity | ].
      split; [ simpl; rewrite Hk2, Hks; reflexivity | ].
      intros [ | i] s'' w w' Hi Hw Hw'; simpl in Hi, Hw, Hw'.
      * intros H. discriminate H.
      * exact (Hrels i s'' w w' Hi Hw Hw').
Qed.

Lemma code_fields_rebuild (id : code_id) (c : code) :
  unchanged id c ->
  code_params (rebuild_c id c) = code_params c
  ∧ code_my_closure (rebuild_c id c) = code_my_closure c
  ∧ code_return_continuation (rebuild_c id c) = code_return_continuation c
  ∧ code_exn_continuation (rebuild_c id c) = code_exn_continuation c
  ∧ code_params_arity (rebuild_c id c) = code_params_arity c
  ∧ code_result_arity (rebuild_c id c) = code_result_arity c
  ∧ code_is_tupled (rebuild_c id c) = code_is_tupled c.
Proof.
  destruct c. unfold unchanged. cbn [metadata_of_code rebuild_code_cc].
  intros Hun. rewrite Hun. repeat split.
Qed.

(* The rebuilt code binds nothing static if the code does not. *)
Lemma rebuilt_code_no_statics (id : code_id) (c : code) :
  checked exact codes (code_body c) = true ->
  binds_statics (code_body c) = false ->
  binds_statics (code_body (rebuild_c id c)) = false.
Proof.
  destruct c as [params mc ret exn md body].
  cbn [code_body rebuild_code_cc].
  intros Hcb Hb.
  destruct (code_calling_convention_change sol cannot_change code_deps
              unit_code_ids id md);
    cbn [code_body];
    (match goal with
     | |- binds_statics (if ?b then _ else _) = false => destruct b
     end; [ reflexivity | ]);
    (match goal with
     | |- binds_statics ?t = false =>
       destruct (binds_statics t) eqn:E; [ | reflexivity ]
     end);
    rewrite (proj1 binds_statics_rebuild body Hcb _ E) in Hb; discriminate Hb.
Qed.

(* The body of the rebuilt code, where the call of the code binds its
   parameters and [my_closure] to values that have sources. *)
Lemma rebuilt_code_body (id : code_id) (c : code) (callee : value)
    (vs : list value)
    (hv : env) :
  unchanged id c ->
  bind_params (code_params c) vs [(code_my_closure c, callee)] = Some hv ->
  env_flows_ hv ->
  code_body (rebuild_c id c)
  = rebuild
      (bind_variables FlambdaReaperRebuild.empty_env
         ((code_my_closure c, Value)
          :: with_kinds (code_params c) (unarize (code_params_arity c))))
      (code_body c).
Proof.
  destruct c as [params mc ret exn md body]. unfold unchanged.
  cbn [code_body rebuild_code_cc code_params code_my_closure
       code_params_arity metadata_of_code].
  intros Hun. rewrite Hun. cbn [code_body].
  intros Hbind Hhv.
  match goal with
  | |- (if ?b then _ else _) = _ => replace b with false
  end; [ reflexivity | symmetry ].
  cbn [existsb].
  rewrite (params_not_dead params _ vs [(mc, callee)] hv Hbind Hhv).
  destruct (bind_params_bound params vs [(mc, callee)] hv mc Hbind)
    as (w & Hw);
    [ right; exists callee; simpl; rewrite Pos.eqb_refl; reflexivity | ].
  pose proof (flows_hs is_local G sol keep called no_loc Hc _ _ (Hhv mc w Hw))
    as Hhs.
  unfold is_dead_var. rewrite Hhs. reflexivity.
Qed.

(* A direct call, in the second run, of a closure of the rebuilt code. *)
Lemma call_rebuilt_code (callee' : value) (kind : call_kind) (id : code_id)
    (slots' : list (value_slot * value)) (c : code) (vs' : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics' : static_env) (hp' : heap) (rcd' : read_record)
    (callee_vars' : env) :
  unchanged id c ->
  closure_of statics' callee' = Some (id, slots') ->
  kind = Indirect ∨ kind = Direct id ->
  lookup id (static_code statics') = Some (rebuild_c id c) ->
  code_is_tupled c = false ->
  length args_arity = length (code_params_arity c) ->
  binds_statics (code_body (rebuild_c id c)) = false ->
  kinds_eqb (map kind_of_value vs') (unarize args_arity) = true ->
  kinds_per_parameter_eqb (unarize_per_parameter args_arity)
    (unarize_per_parameter (code_params_arity c)) = true ->
  arity_eqb return_arity (code_result_arity c) = true ->
  bind_params (code_params c) vs' [(code_my_closure c, callee')]
  = Some callee_vars' ->
  call_function callee' kind vs' args_arity return_arity return_to exn_to
    statics' hp' rcd'
  = Step (State (code_body (rebuild_c id c)) callee_vars'
            [(code_return_continuation c, return_to);
             (code_exn_continuation c, exn_to)]
            statics' hp' rcd').
Proof.
  intros Hun Hcl Hk Hcode Htup Hfull Hbinds Hk1 Hk2 Hk3 Hbind.
  destruct (code_fields_rebuild id c Hun)
    as (Ep & Em & Er & Ee & Epa & Era & Et).
  assert (Hshape : shape_of_call statics' callee' kind vs' args_arity
                     return_arity
                   = Some (Full_call callee' (rebuild_c id c) vs' args_arity)).
  { assert (Hfull' :
      shape_of_closure_call statics' callee' kind vs' args_arity return_arity
      = Some (Full_call callee' (rebuild_c id c) vs' args_arity)).
    { apply (shape_of_closure_call_full _ _ _ _ _ _ id slots' _ Hcl Hk Hcode);
        [ rewrite Et; exact Htup | rewrite Epa; exact Hfull ]. }
    destruct Hk as [-> | ->]; [ | exact Hfull' ].
    unfold shape_of_call.
    destruct callee'; try discriminate Hcl; exact Hfull'. }
  unfold call_function. rewrite Hshape. cbn [run_call]. unfold call_code.
  rewrite Hbinds, Hk1, Epa, Hk2, Era, Hk3. cbn [negb andb].
  rewrite Ep, Em, Hbind, Er, Ee. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Code that changes its calling convention                                 *)
(* ------------------------------------------------------------------------ *)

(* The code [id] changes its calling convention, with the decisions [pd]
   of its parameters and [rd] of its results. *)
Definition changing (id : code_id) (c : code)
    (pd rd : list param_decision) : Prop :=
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    (metadata_of_code c)
  = Changing_calling_convention pd rd.

Lemma code_fields_rebuild_cc (id : code_id) (c : code)
    (pd rd : list param_decision) :
  changing id c pd rd ->
  code_params (rebuild_c id c) = filter_kept (map is_kept pd) (code_params c)
  ∧ code_my_closure (rebuild_c id c) = code_my_closure c
  ∧ code_return_continuation (rebuild_c id c) = code_return_continuation c
  ∧ code_exn_continuation (rebuild_c id c) = code_exn_continuation c
  ∧ code_params_arity (rebuild_c id c)
    = arity_of_grouped_decisions
        (group_by_parameter (code_params_arity c) pd)
  ∧ code_result_arity (rebuild_c id c) = arity_of_decisions rd
  ∧ code_is_tupled (rebuild_c id c) = false.
Proof.
  destruct c. unfold changing. cbn [metadata_of_code rebuild_code_cc].
  intros Hch. rewrite Hch. repeat split.
Qed.

(* The body of the rebuilt code, in which the return continuation keeps
   the results kept. *)
Lemma rebuilt_code_body_cc (id : code_id) (c : code)
    (pd rd : list param_decision) (callee : value) (vs : list value)
    (hv : env) :
  changing id c pd rd ->
  bind_params (code_params c) vs [(code_my_closure c, callee)] = Some hv ->
  env_flows_ hv ->
  code_body (rebuild_c id c)
  = rebuild
      (bind_variables
         (bind_continuations FlambdaReaperRebuild.empty_env
            [(code_return_continuation c, map is_kept rd)])
         ((code_my_closure c, Value)
          :: with_kinds (code_params c) (unarize (code_params_arity c))))
      (code_body c).
Proof.
  destruct c as [params mc ret exn md body]. unfold changing.
  cbn [code_body rebuild_code_cc code_params code_my_closure
       code_params_arity code_return_continuation metadata_of_code].
  intros Hch. rewrite Hch. cbn [code_body].
  intros Hbind Hhv.
  match goal with
  | |- (if ?b then _ else _) = _ => replace b with false
  end; [ reflexivity | symmetry ].
  cbn [existsb].
  rewrite (params_not_dead params _ vs [(mc, callee)] hv Hbind Hhv).
  destruct (bind_params_bound params vs [(mc, callee)] hv mc Hbind)
    as (w & Hw);
    [ right; exists callee; simpl; rewrite Pos.eqb_refl; reflexivity | ].
  pose proof (flows_hs is_local G sol keep called no_loc Hc _ _ (Hhv mc w Hw))
    as Hhs.
  unfold is_dead_var. rewrite Hhs. reflexivity.
Qed.

(* A direct call, in the second run, of a closure of the rebuilt code
   that changes its calling convention, at its arities. *)
Lemma call_rebuilt_code_cc (callee' : value) (kind : call_kind)
    (id : code_id) (slots' : list (value_slot * value)) (c : code)
    (pd rd : list param_decision) (vs' : list value)
    (return_to exn_to : cont_value) (statics' : static_env) (hp' : heap)
    (rcd' : read_record)
    (callee_vars' : env) :
  changing id c pd rd ->
  closure_of statics' callee' = Some (id, slots') ->
  kind = Indirect ∨ kind = Direct id ->
  lookup id (static_code statics') = Some (rebuild_c id c) ->
  binds_statics (code_body (rebuild_c id c)) = false ->
  kinds_eqb (map kind_of_value vs')
    (unarize (code_params_arity (rebuild_c id c)))
  = true ->
  bind_params (filter_kept (map is_kept pd) (code_params c)) vs'
    [(code_my_closure c, callee')]
  = Some callee_vars' ->
  call_function callee' kind vs' (code_params_arity (rebuild_c id c))
    (arity_of_decisions rd) return_to exn_to statics' hp' rcd'
  = Step (State (code_body (rebuild_c id c)) callee_vars'
            [(code_return_continuation c, return_to);
             (code_exn_continuation c, exn_to)]
            statics' hp' rcd').
Proof.
  intros Hch Hcl Hk Hcode Hbinds Hk1 Hbind.
  destruct (code_fields_rebuild_cc id c pd rd Hch)
    as (Ep & Em & Er & Ee & Epa & Era & Et).
  set (A := code_params_arity (rebuild_c id c)) in *.
  assert (Hshape : shape_of_call statics' callee' kind vs' A
                     (arity_of_decisions rd)
                   = Some (Full_call callee' (rebuild_c id c) vs' A)).
  { assert (Hfull' :
      shape_of_closure_call statics' callee' kind vs' A (arity_of_decisions rd)
      = Some (Full_call callee' (rebuild_c id c) vs' A))
      by exact (shape_of_closure_call_full _ _ _ _ _ _ id slots' _ Hcl Hk
                  Hcode Et eq_refl).
    destruct Hk as [-> | ->]; [ | exact Hfull' ].
    unfold shape_of_call.
    destruct callee'; try discriminate Hcl; exact Hfull'. }
  unfold call_function. rewrite Hshape. cbn [run_call]. unfold call_code.
  rewrite Hbinds, Hk1. fold A.
  replace (kinds_per_parameter_eqb (unarize_per_parameter A)
             (unarize_per_parameter A)) with true
    by (symmetry; apply kinds_per_parameter_eqb_eq; reflexivity).
  rewrite Era. replace (arity_eqb (arity_of_decisions rd)
                          (arity_of_decisions rd)) with true
    by (symmetry; apply arity_eqb_eq; reflexivity).
  cbn [negb andb].
  rewrite Ep, Em, Hbind, Er, Ee. reflexivity.
Qed.

(* The arguments and the arity of a call that changes its calling
   convention ([rebuild_apply_cc]), at the arity of the code: the
   arguments kept, at the arity of the rebuilt code. *)
Lemma cc_call_args (en : rebuild_env) (ap : apply_expr) (md : code_metadata)
    (pd : list param_decision) :
  unarize_per_parameter (ap_args_arity ap)
  = unarize_per_parameter (cm_params_arity md) ->
  length (ap_args ap) = length (unarize (ap_args_arity ap)) ->
  length pd = length (unarize (cm_params_arity md)) ->
  let groups :=
    map (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
      (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
         (group_by_parameter (ap_args_arity ap) pd))
  in
  flat_map (map fst) groups
  = filter_kept (map is_kept pd) (map (rewrite_simple sol en) (ap_args ap))
  ∧ arity_of_groups groups
    = arity_of_grouped_decisions (group_by_parameter (cm_params_arity md) pd).
Proof.
  intros Hk2 La Lpd groups.
  assert (Hs : list_sum (arity_sizes (ap_args_arity ap))
               = length (unarize (ap_args_arity ap)))
    by apply sum_arity_sizes.
  assert (Hs' : arity_sizes (ap_args_arity ap)
                = arity_sizes (cm_params_arity md))
    by (unfold arity_sizes; rewrite Hk2; reflexivity).
  assert (Lpd' : length pd = list_sum (arity_sizes (ap_args_arity ap))).
  { rewrite Hs, Lpd, (unarize_per_parameter_eq _ _ Hk2). reflexivity. }
  split.
  - exact (cc_groups_args sol en (arity_sizes (ap_args_arity ap)) _ _
             (eq_trans La (eq_sym Hs)) Lpd').
  - unfold group_by_parameter. fold (arity_sizes (cm_params_arity md)).
    rewrite <- Hs'.
    exact (cc_groups_arity sol en (arity_sizes (ap_args_arity ap)) _ _
             (eq_trans La (eq_sym Hs)) Lpd').
Qed.

(* What a call reads of the witness of the code, related at a node of
   the call, is related at the node of the code. *)
Lemma outer_nodes_read (k0 : continuation) (r r' w witness : node)
    (f : field) :
  is_local f = false -> g_constructor G w f r' -> g_accessor G r f witness ->
  may_come_from sol w witness -> outer_nodes k0 [r] -> outer_nodes k0 [r'].
Proof.
  intros Hl Hcons Hacc Hw [Hlen Hon]. split; [ exact Hlen | ].
  intros i n p Hn Hp. destruct i as [ | i]; [ | destruct i; discriminate Hn ].
  injection Hn as <-.
  destruct (Hon 0 r p eq_refl Hp) as [Hhu Hrel]. split.
  - exact (read_hu _ _ _ _ _ Hl Hcons Hacc Hw Hhu).
  - intros v v' Hv. apply Hrel.
    exact (rel_read G sol keep called (exact = true) True is_local Hc
             _ _ _ _ _ v v' Hl Hcons Hacc Hw Hv).
Qed.

(* The values of the parameters of the code, as [rel_argument]. *)
Lemma rel_argument (w base to_ from : node) (i : nat) (v v' : value) :
  may_come_from sol w base -> g_argument G from i base ->
  g_parameter G w i to_ -> hs from -> rel v from v' -> rel v to_ v'.
Proof.
  intros Hw Harg Hpar Hhs.
  apply (argument_transfer _ _ _ Hc (fun n => rel v n v') w base to_ from i);
    [ | | exact Hw | exact Harg | exact Hpar | exact Hhs ].
  - intros x y. exact (rel_alias G sol keep called (exact = true) True
      is_local Hc
                         v x y v').
  - intros x m. exact (rel_escape G sol keep called (exact = true) True
      is_local Hc
                         v x m v').
Qed.

(* What a call reads of the witness of the code, at a node of the code,
   is related at the node of the call. *)
Lemma read_transports (r r' w witness : node) (f : field) :
  is_local f = false -> g_constructor G w f r' -> g_accessor G r f witness ->
  may_come_from sol w witness -> transports r' r.
Proof.
  intros Hl Hcons Hacc Hw. split.
  - exact (read_hu _ _ _ _ _ Hl Hcons Hacc Hw).
  - intros v v' Hv.
    exact (rel_read G sol keep called (exact = true) True is_local Hc
             _ _ _ _ _ v v' Hl Hcons Hacc Hw Hv).
Qed.

(* A way out at the nodes of a call is one at the nodes of the code. *)
Lemma way_out_transport (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (ns ns' : list node) (c c' : cont_value) :
  way_out R R2 ns c c' -> length ns' = length ns ->
  (∀ (i : nat) (n n' : node),
     nth_error ns i = Some n -> nth_error ns' i = Some n' -> transports n' n) ->
  way_out R R2 ns' c c'.
Proof.
  intros Hway Hlen Htr.
  assert (Hat : ∀ (i : nat) (n' : node),
            nth_error ns' i = Some n' ->
            ∃ n : node, nth_error ns i = Some n ∧ transports n' n).
  { intros i n' Hn'. destruct (nth_error ns i) as [n | ] eqn:Hn.
    - exists n. split; [ reflexivity | exact (Htr i n n' Hn Hn') ].
    - exfalso. apply nth_error_None in Hn.
      assert (i < length ns') by (apply nth_error_Some; congruence). lia. }
  destruct Hway as [(k0 & Hc0 & Hc0' & Hunit0 & Hlen0 & Hon)
                   | [(mask & c'' & Hret & HR & Hc')
                     | [(n0 & mask & h & h' & extras & extras' & -> & Hc0
                         & Hc0' & Hret & HR & Hhd & Hx)
                       | (n & n0 & ns0 & -> & [Hhu0 Hrel0] & HR2)]]].
  - left. exists k0.
    split; [ exact Hc0 | split; [ exact Hc0' | split; [ exact Hunit0 | ] ] ].
    split; [ rewrite Hlen; exact Hlen0 | ].
    intros i n' p Hn' Hp. destruct (Hat i n' Hn') as (n & Hn & [Hhu Hrel]).
    destruct (Hon i n p Hn Hp) as [Hhu_n Hrel_n].
    split; [ exact (Hhu Hhu_n) | ].
    intros v v' Hv. exact (Hrel_n v v' (Hrel v v' Hv)).
  - right. left. exists mask, c''. split; [ | exact (conj HR Hc') ].
    intros i n' p Hn' Hp. destruct (Hat i n' Hn') as (n & Hn & [Hhu Hrel]).
    destruct (Hret i n p Hn Hp) as [Hhu_n Hrel_n].
    split; [ intros Hp'; exact (Hhu (Hhu_n Hp')) | ].
    intros v v' Hv. exact (Hrel_n v v' (Hrel v v' Hv)).
  - right. right. left.
    destruct ns' as [ | n' [ | n'' ns']]; try discriminate Hlen.
    exists n', mask, h, h', extras, extras'.
    split; [ reflexivity | split; [ exact Hc0 | split; [ exact Hc0' | ] ] ].
    split; [ | exact (conj HR (conj Hhd Hx)) ].
    intros i m p Hm Hp. destruct (Hat i m Hm) as (m0 & Hm0 & [Hhu Hrel]).
    destruct (Hret i m0 p Hm0 Hp) as [Hhu_n Hrel_n].
    split; [ intros Hp'; exact (Hhu (Hhu_n Hp')) | ].
    intros v v' Hv. exact (Hrel_n v v' (Hrel v v' Hv)).
  - right. right. right.
    destruct ns' as [ | n' ns']; [ discriminate Hlen | ].
    exists n', n0, ns'. split; [ reflexivity | split; [ | exact HR2 ] ].
    destruct (Hat 0 n' eq_refl) as (n1 & Hn1 & [Hhu Hrel]).
    simpl in Hn1. injection Hn1 as <-.
    split; [ intros H; exact (Hhu (Hhu0 H)) | ].
    intros v v' Hv. exact (Hrel0 v v' (Hrel v v' Hv)).
Qed.

Lemma transports_trans (a b c : node) :
  transports a b -> transports b c -> transports a c.
Proof.
  intros [H1 H2] [H3 H4]. split.
  - intros Hcu. exact (H1 (H3 Hcu)).
  - intros v v' Hv. exact (H4 v v' (H2 v v' Hv)).
Qed.

(* The return continuation of code at other nodes, to which those of the
   code transport what they have: a call in tail position, which
   returns there with the results of the code it runs. *)
Lemma ret_rel_transport (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (m : list bool) (K : list flambda_kind) (ns ns' : list node)
    (c c' : cont_value) :
  ret_rel R R2 m K ns c c' -> length ns' = length ns ->
  (∀ (i : nat) (n n' : node),
     nth_error ns i = Some n -> nth_error ns' i = Some n' -> transports n' n) ->
  ret_rel R R2 m K ns' c c'.
Proof.
  intros Hrr.
  destruct Hrr as [m K ns c c' Hrr
                  | m0 K ns ns0 c k rd vars conts Hin Lk L2 Lns Hex Htr0 Hkeep
                      Hdel];
    intros Hlen Htr.
  2: { apply (Ret_wrapper_ret R R2 m0 K ns' ns0 c k rd vars conts);
         [ exact Hin | exact Lk | rewrite Hlen; exact L2
         | rewrite Hlen; exact Lns | exact Hex | | exact Hkeep
         | exact Hdel ].
       intros i n'' n' Hn'' Hn'. destruct (nth_error ns i) as [n | ] eqn:Hn.
       - exact (transports_trans _ _ _ (Htr i n n'' Hn Hn'')
                  (Htr0 i n n' Hn Hn')).
       - exfalso. apply nth_error_None in Hn.
         assert (i < length ns') by (apply nth_error_Some; congruence).
         lia. }
  apply Ret_base.
  assert (Hat : ∀ (i : nat) (n' : node),
            nth_error ns' i = Some n' ->
            ∃ n : node, nth_error ns i = Some n ∧ transports n' n).
  { intros i n' Hn'. destruct (nth_error ns i) as [n | ] eqn:Hn.
    - exists n. split; [ reflexivity | exact (Htr i n n' Hn Hn') ].
    - exfalso. apply nth_error_None in Hn.
      assert (i < length ns') by (apply nth_error_Some; congruence). lia. }
  assert (Hret : ∀ c0 : cont_value, returns_to ns c0 -> returns_to ns' c0).
  { intros c0 Hr i n' p Hn' Hp.
    destruct (Hat i n' Hn') as (n & Hn & [Hhu Hrel]).
    destruct (Hr i n p Hn Hp) as [Hhu' Hrel'].
    split; [ intros Hpu; exact (Hhu (Hhu' Hpu)) | ].
    intros v v' Hv. exact (Hrel' v v' (Hrel v v' Hv)). }
  destruct Hrr as [[Hall Hway] | [(hm & Hr & HR & ->)
    | [(k & hm & rd & kinds & vars & conts & c'' & E1 & E2 & Hr & HR & E3)
      | (k & rd & vars & conts & nsw & E1 & Hway & E3 & E4 & E5 & E5b & E6
         & E7 & E8)]]].
  - left. split; [ exact Hall | ].
    exact (way_out_transport R R2 ns ns' c c' Hway Hlen Htr).
  - right. left. exists hm.
    split; [ exact (Hret c Hr) | split; [ exact HR | rewrite Hlen;
                                          reflexivity ] ].
  - right. right. left. exists k, hm, rd, kinds, vars, conts, c''.
    split; [ exact E1 | split; [ exact E2 | ] ].
    split; [ exact (Hret c Hr) | split; [ exact HR | exact E3 ] ].
  - right. right. right. exists k, rd, vars, conts, nsw.
    split; [ exact E1 | split; [ exact Hway | split; [ exact E3 | ] ] ].
    split; [ exact E4 | split; [ rewrite Hlen; exact E5 | ] ].
    split; [ rewrite Hlen; exact E5b | ].
    split; [ exact E6 | split; [ | exact E8 ] ].
    intros i n'' n' Hn'' Hn'.
    destruct (Hat i n'' Hn'') as (n & Hn & Htr').
    exact (transports_trans _ _ _ Htr' (E7 i n n' Hn Hn')).
Qed.

Lemma forallb_false_nth (l : list bool) :
  forallb (fun keep => keep) l = false ->
  ∃ i : nat, nth_error l i = Some false.
Proof.
  induction l as [ | b l IH]; [ discriminate | ].
  cbn [forallb]. destruct b; cbn [andb].
  - intros H. destruct (IH H) as (i & Hi). exists (S i). exact Hi.
  - intros _. exists 0. reflexivity.
Qed.

Lemma forallb_true_repeat (l : list bool) :
  forallb (fun keep => keep) l = true -> l = repeat true (length l).
Proof.
  induction l as [ | b l IH]; [ reflexivity | ].
  cbn [forallb]. intros H. apply andb_true_iff in H as [-> H].
  cbn [length repeat]. f_equal. exact (IH H).
Qed.

Lemma kept_kinds_keep (K : list flambda_kind) : kept_kinds (map Keep K) = K.
Proof.
  induction K as [ | k K IH]; [ reflexivity | ].
  unfold kept_kinds in *. cbn [map flat_map app]. rewrite IH. reflexivity.
Qed.

Lemma map_fst_combine_keep (K : list flambda_kind) :
  map fst (combine (map Keep K) K) = map Keep K.
Proof.
  induction K as [ | k K IH]; [ reflexivity | ].
  cbn [map combine fst]. rewrite IH. reflexivity.
Qed.

Lemma map_is_kept_keep (K : list flambda_kind) :
  map is_kept (map Keep K) = repeat true (length K).
Proof.
  induction K as [ | k K IH]; [ reflexivity | ].
  cbn [map is_kept length repeat]. rewrite IH. reflexivity.
Qed.

Lemma wrapper_args_keep (K : list flambda_kind) :
  ∀ (next : positive) (m : list bool),
    length m = length K ->
    wrapper_args next m (combine (map Keep K) K)
    = filter_kept m (map Var (names_from (length K) next)).
Proof.
  induction K as [ | k K IH]; intros next [ | b m] L; try discriminate L;
    [ reflexivity | ].
  injection L as L.
  cbn [map combine wrapper_args length names_from filter_kept].
  rewrite (IH (Pos.succ next) m L). destruct b; reflexivity.
Qed.

(* The wrapper of [make_apply_wrapper] is that of [make_apply_wrapper_cc]
   for code that keeps every result. *)
Lemma wrapper_as_cc (k : continuation) (m : list bool)
    (K : list flambda_kind) (vars : env) (conts : cont_env) :
  length m = length K ->
  wrapper k m K vars conts
  = cc_wrapper k (pad_mask (length (map Keep K)) m) (combine (map Keep K) K)
      vars conts.
Proof.
  intros L. unfold wrapper, cc_wrapper.
  rewrite map_fst_combine_keep, kept_kinds_keep, length_map, <- L,
    pad_mask_self, L, wrapper_args_keep by exact L.
  reflexivity.
Qed.

(* The plain wrapper of a call in tail position over the return
   continuation of code that changes, which keeps fewer results. *)
Lemma ret_rel_plain_wrapper
    (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (m : list bool) (K : list flambda_kind) (ns ns0 : list node)
    (c : cont_value) (k : continuation) (vars : env) (conts : cont_env) :
  ret_rel R R2 m K ns0 c (resolve k conts) ->
  length m = length K -> length ns0 = length ns -> length ns = length K ->
  forallb (fun keep => keep) m = false ->
  (∀ (i : nat) (n n' : node),
     nth_error ns i = Some n -> nth_error ns0 i = Some n' ->
     transports n n') ->
  ret_rel R R2 (repeat true (length K)) K ns c (wrapper k m K vars conts).
Proof.
  intros Hin Lm L0 Lns Hf Htr.
  destruct (forallb_false_nth m Hf) as (i & Hi).
  rewrite (wrapper_as_cc k m K vars conts Lm), <- map_is_kept_keep.
  apply (Ret_wrapper_ret R R2 m K ns ns0 c k (map Keep K) vars conts Hin).
  - apply length_map.
  - exact L0.
  - rewrite length_map. exact Lns.
  - exists i. left. exact Hi.
  - exact Htr.
  - intros j kind Hj. rewrite nth_error_map in Hj.
    destruct (nth_error K j); cbn [option_map] in Hj; [ | discriminate Hj ].
    injection Hj as <-. reflexivity.
  - intros j Hj. exfalso. rewrite nth_error_map in Hj.
    destruct (nth_error K j); discriminate Hj.
Qed.

(* The return continuation of code that changes, which keeps every
   result, at other nodes. *)
Lemma ret_rel_all_true (R : list bool -> cont_value -> cont_value -> Prop)
    (R2 : node -> cont_value -> cont_value -> Prop)
    (m : list bool) (K : list flambda_kind) (ns ns' : list node)
    (c c' : cont_value) :
  ret_rel R R2 m K ns c c' -> forallb (fun keep => keep) m = true ->
  length m = length K -> length ns' = length ns ->
  (∀ (i : nat) (n n' : node),
     nth_error ns i = Some n -> nth_error ns' i = Some n' -> transports n' n) ->
  ret_rel R R2 (repeat true (length K)) K ns' c c'.
Proof.
  intros Hrr Hall Lm Lns Htr.
  rewrite <- Lm, <- (forallb_true_repeat m Hall).
  exact (ret_rel_transport R R2 m K ns ns' c c' Hrr Lns Htr).
Qed.

(* The continuations that the rebuild of a call names, wrapper or not. *)
Lemma make_apply_wrapper_free (en : rebuild_env) (ap : apply_expr) :
  ap_exn_continuation ap <> first_fresh ->
  ap_continuation ap
    ∈ free_continuations_of_expr (make_apply_wrapper first_fresh en ap)
  ∧ ap_exn_continuation ap
    ∈ free_continuations_of_expr (make_apply_wrapper first_fresh en ap).
Proof.
  intros Hne. unfold make_apply_wrapper.
  destruct (lookup (ap_continuation ap) (env_conts en)) as [mask | ];
    [ destruct (forallb (fun keep => keep) mask) | ];
    cbn [free_continuations_of_expr free_continuations_of_handler
         apply_cont_continuations ac_continuation ac_trap_action
         with_continuation ap_continuation ap_exn_continuation];
    try (split; [ left; reflexivity | right; left; reflexivity ]).
  split; [ left; reflexivity | ].
  right. unfold without. apply filter_In.
  split; [ right; left; reflexivity | ].
  rewrite among_cons. apply Pos.eqb_neq in Hne. rewrite Hne. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls of unknown arity                                                   *)
(* ------------------------------------------------------------------------ *)

Abbreviation wchain := (witness_chain G sol).

Lemma wchain_mono (P P' : node -> Prop) :
  (∀ y : node, P y -> P' y) ->
  ∀ (g : nat) (W : node), wchain P g W -> wchain P' g W.
Proof.
  intros HP g. induction g as [ | g IH]; intros W; cbn [witness_chain].
  - intros (y & Hy & Hcons & HW). exists y. auto.
  - intros (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HW).
    exists W', R, v. split; [ exact (IH W' HW') | auto ].
Qed.

(* What the first results of the witnesses of the first group may come
   from. *)
Definition wchain_next (P : node -> Prop) (v : node) : Prop :=
  ∃ W0 R : node,
    wchain P 0 W0
    ∧ g_constructor G W0 (Field_return_of_call (Normal 0)) R
    ∧ may_come_from sol v R
    ∧ may_come_from sol v v.

Lemma wchain_shift (P : node -> Prop) :
  ∀ (g : nat) (W : node), wchain P (S g) W -> wchain (wchain_next P) g W.
Proof.
  induction g as [ | g IH]; intros W.
  - intros (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HW).
    exists v. split; [ exists W', R; auto | auto ].
  - intros (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HW).
    exists W', R, v. split; [ exact (IH W' HW') | auto ].
Qed.

(* A witness that the group [g] of the arguments of a call reaches may
   come to the witness of the call of that group. *)
Lemma wchain_cover :
  ∀ (g : nat) (P : node -> Prop) (ws : list node) (W w : node),
    call_chain G sol P ws -> wchain P g W -> nth_error ws g = Some w ->
    may_come_from sol W w.
Proof.
  induction g as [ | g IH]; intros P ws W w Hchain HW Hw;
    (destruct ws as [ | w0 ws]; [ discriminate Hw | ]);
    destruct Hchain as [Hcalled Hchain].
  - simpl in Hw. injection Hw as <-. destruct HW as (y & Hy & Hcons & HWW).
    exact (Hcalled y W Hy Hcons HWW).
  - simpl in Hw.
    apply (IH (returned_at G sol w0) ws W w Hchain); [ | exact Hw ].
    apply (wchain_mono (wchain_next P)); [ | exact (wchain_shift P g W HW) ].
    intros v (W0 & R & HW0 & HR & HvR & Hvv). exists W0, R.
    destruct HW0 as (y & Hy & Hcons & HW0W).
    split; [ exact (Hcalled y W0 Hy Hcons HW0W) | auto ].
Qed.

(* The witnesses of unknown arity of code that is not tupled are those
   that its groups of parameters reach from the allocation of a closure
   of the code. *)
Lemma code_wchain (id : code_id) (returns : list node) (exn : node)
    (L : list (list node * node)) (P : node -> Prop) :
  code_chain_edges G id returns exn L ->
  hs (Node_code_id id) ->
  (∀ (first : list node) (W : node),
     hd_error L = Some (first, W) ->
     ∃ y : node,
       P y
       ∧ g_constructor G y (Field_call_witness Unknown_arity_code_pointer) W)
  ->
  ∀ (g : nat) (first : list node) (W : node),
    nth_error L g = Some (first, W) -> wchain P g W.
Proof.
  intros Hch Hhs Hhd g. induction g as [ | g IH]; intros first W Hg.
  - destruct (code_chain_nth G _ _ _ _ _ _ _ Hch Hg) as (_ & Hcode & _).
    assert (Hhd' : hd_error L = Some (first, W))
      by (destruct L; [ discriminate Hg | exact Hg ]).
    destruct (Hhd first W Hhd') as (y & Hy & Hcons).
    exists y. split; [ exact Hy | split; [ exact Hcons | ] ].
    exact (alloc_self is_local G sol Hc _ _ _ Hcode Hhs).
  - assert (Hg' : ∃ (f : list node) (W' : node),
              nth_error L g = Some (f, W')).
    { destruct (nth_error L g) as [[f W'] | ] eqn:E; [ eauto | ].
      apply nth_error_None in E.
      assert (S g < length L) by (apply nth_error_Some; congruence). lia. }
    destruct Hg' as (f & W' & HW').
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hch HW') as (_ & _ & _ & Hnext).
    destruct (Hnext first W Hg) as (v & Hv & Hvw).
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hch Hg) as (_ & Hcode & _).
    pose proof (alloc_self is_local G sol Hc _ _ _ Hcode Hhs) as HWW.
    pose proof (alloc_self is_local G sol Hc _ _ _ Hvw (proj1 HWW)) as Hvv.
    exists W', v, v. split; [ exact (IH f W' HW') | ].
    split; [ exact Hv | split; [ exact Hvv | split; [ exact Hvv | ] ] ].
    split; [ exact Hvw | exact HWW ].
Qed.

Lemma all_kept_true (sizes : list nat) (g k : nat) (m : list bool) :
  nth_error (map (fun n => repeat true n) sizes) g = Some m ->
  k < length m -> nth_error m k = Some true.
Proof.
  intros Hg Hk. rewrite nth_error_map in Hg.
  destruct (nth_error sizes g) as [n | ]; [ | discriminate Hg ].
  injection Hg as <-. rewrite repeat_length in Hk.
  exact (nth_error_repeat_lt true _ _ Hk).
Qed.

(* A mask for each group, of the size of the group. *)
Lemma arguments_used_lengths (ep : closure_entry_point) :
  ∀ (sizes : list nat) (src : allocation_sources),
    Forall2 (fun m n => length m = n)
      (arguments_used_by_call F sol ep src sizes) sizes.
Proof.
  assert (Hall : ∀ sizes : list nat,
            Forall2 (fun m n => length m = n)
              (map (fun n => repeat true n) sizes) sizes).
  { induction sizes as [ | n sizes IH]; constructor;
      [ apply repeat_length | exact IH ]. }
  induction sizes as [ | n sizes IH]; intros src; [ constructor | ].
  cbn [arguments_used_by_call].
  destruct src as [ | Ss]; [ exact (Hall (n :: sizes)) | ].
  destruct (get_field_sources F sol Ss (Field_call_witness ep)) as [ | WS];
    [ exact (Hall (n :: sizes)) | ].
  constructor; [ rewrite length_map; apply length_seq | ].
  destruct sizes as [ | n' sizes]; [ constructor | apply IH ].
Qed.

(* The masks of the groups, flattened: the argument at [i] is in the
   group of [group_position]. *)
Lemma concat_group_nth {A : Type} (sizes : list nat) (ms : list (list A)) :
  Forall2 (fun m n => length m = n) ms sizes ->
  ∀ i g k : nat,
    group_position sizes i = Some (g, k) ->
    ∃ m : list A,
      nth_error ms g = Some m ∧ k < length m
      ∧ nth_error (concat ms) i = nth_error m k.
Proof.
  induction 1 as [ | m n ms sizes Hmn _ IH]; intros i g k Hpos;
    simpl in Hpos; [ discriminate Hpos | ].
  destruct (Nat.ltb_spec i n) as [Hi | Hi].
  - injection Hpos as <- <-. exists m. simpl.
    split; [ reflexivity | split; [ lia | ] ].
    apply nth_error_app1. lia.
  - destruct (group_position sizes (i - n)) as [[g' k'] | ] eqn:E;
      [ | discriminate Hpos ].
    injection Hpos as <- <-.
    destruct (IH _ _ _ E) as (m' & Hm' & Hk & Hnth).
    exists m'. simpl. split; [ exact Hm' | split; [ exact Hk | ] ].
    rewrite nth_error_app2 by lia. rewrite Hmn. exact Hnth.
Qed.

Lemma concat_length_sizes {A : Type} (sizes : list nat)
    (ms : list (list A)) :
  Forall2 (fun m n => length m = n) ms sizes ->
  length (concat ms) = list_sum sizes.
Proof.
  induction 1 as [ | m n ms sizes Hmn _ IH]; simpl; [ reflexivity | ].
  rewrite length_app, Hmn, IH. reflexivity.
Qed.

(* The mask of the arguments of a call of unknown arity, group by
   group. *)
Lemma unknown_mask_eq (callee : node) (a : arity) :
  arguments_used_by_unknown_arity_call F sol callee a
  = concat
      (arguments_used_by_call F sol Unknown_arity_code_pointer
         (get_direct_sources sol callee) (arity_sizes a)).
Proof.
  unfold arguments_used_by_unknown_arity_call, arity_sizes,
    unarize_per_parameter.
  rewrite map_map. reflexivity.
Qed.

(* What [get_field_sources] finds. *)
Lemma field_sources_cases (in_ : node -> bool) (f : field) :
  (get_field_sources F sol in_ f = Any_source
   ∧ ∃ x y : node,
       (x, f, y) ∈ f_constructor F ∧ in_ x = true ∧ as_ y)
  ∨ ∃ WS : node -> bool,
      get_field_sources F sol in_ f = Sources WS
      ∧ (∀ x y : node,
           (x, f, y) ∈ f_constructor F -> in_ x = true ->
           s_any_source sol y = false)
      ∧ ∀ z : node,
          WS z = true
          <-> ∃ x y : node,
                (x, f, y) ∈ f_constructor F ∧ in_ x = true
                ∧ s_sources sol y z = true ∧ hs z.
Proof.
  unfold get_field_sources.
  destruct (existsb _ (f_constructor F)) eqn:Hex.
  - left. split; [ reflexivity | ].
    apply existsb_exists in Hex as ([[x f'] y] & Hin & H).
    apply andb_true_iff in H as [H Hy]. apply andb_true_iff in H as [Hx Hf].
    apply field_eqb_true in Hf. subst f'. exists x, y. auto.
  - right. eexists. split; [ reflexivity | split ].
    + intros x y Hin Hx.
      destruct (s_any_source sol y) eqn:Ey; [ | reflexivity ].
      apply not_true_iff_false in Hex. exfalso. apply Hex.
      apply existsb_exists. exists (x, f, y). split; [ exact Hin | ].
      rewrite Hx, Ey, (proj2 (field_eqb_true _ _) eq_refl). reflexivity.
    + intros z. cbv beta. split.
      * intros Hz. apply existsb_exists in Hz as ([[x f'] y] & Hin & H).
        apply andb_true_iff in H as [H Hhz].
        apply andb_true_iff in H as [H Hyz].
        apply andb_true_iff in H as [Hx Hf]. apply field_eqb_true in Hf.
        subst f'. exists x, y. auto.
      * intros (x & y & Hin & Hx & Hyz & Hhz). apply existsb_exists.
        exists (x, f, y). split; [ exact Hin | ].
        rewrite Hx, Hyz, Hhz, (proj2 (field_eqb_true _ _) eq_refl).
        reflexivity.
Qed.

(* The allocations that [P] holds of are among the sources [src]. *)
Definition src_has (src : allocation_sources) (y : node) : Prop :=
  match src with
  | Any_source => True
  | Sources Ss => Ss y = true
  end.

(* A parameter that has a use, of a witness that the group [g] of the
   arguments of a call of unknown arity reaches from what the callee may
   come from, has its argument kept. *)
Lemma unknown_mask_used :
  ∀ (sizes : list nat) (src : allocation_sources) (P : node -> Prop),
    (∀ y : node, P y -> src_has src y) ->
    ∀ (g k : nat) (W p : node) (m : list bool),
      wchain P g W ->
      nth_error
        (arguments_used_by_call F sol Unknown_arity_code_pointer src sizes) g
      = Some m ->
      g_parameter G W k p -> hu p -> k < length m ->
      nth_error m k = Some true.
Proof.
  induction sizes as [ | n sizes IH];
    intros src P HP g k W p m HW Hm Hpar Hp Hk;
    [ destruct g; discriminate Hm | ].
  cbn [arguments_used_by_call] in Hm.
  destruct src as [ | Ss]; [ exact (all_kept_true _ _ _ _ Hm Hk) | ].
  destruct (field_sources_cases Ss
              (Field_call_witness Unknown_arity_code_pointer))
    as [[Hany _] | (WS & Hws & Hnot & Hiff)].
  { rewrite Hany in Hm. exact (all_kept_true _ _ _ _ Hm Hk). }
  rewrite Hws in Hm.
  (* The witnesses of the first group are among [WS]. *)
  assert (HWS : ∀ W0 : node, wchain P 0 W0 -> WS W0 = true).
  { intros W0 (y & Hy & Hcons & (Hhs & Hsrc & _)).
    apply Hiff. exists y, W0.
    split; [ exact (constructor_fact _ _ _ Hcons) | ].
    split; [ exact (HP y Hy) | split; [ | exact Hhs ] ].
    destruct Hsrc as [Has | Hs]; [ | exact Hs ].
    rewrite (Hnot y W0 (constructor_fact _ _ _ Hcons) (HP y Hy)) in Has.
    discriminate Has. }
  destruct g as [ | g].
  - cbn [nth_error] in Hm. injection Hm as <-.
    rewrite length_map, length_seq in Hk.
    rewrite nth_error_map, (nth_error_seq_lt 0 _ _ Hk). simpl. f_equal.
    unfold cofield_has_use. apply existsb_exists.
    exists (W, k, p). split; [ exact (parameter_fact _ _ _ Hpar) | ].
    rewrite (HWS W HW), Nat.eqb_refl, Hp. reflexivity.
  - destruct sizes as [ | n' sizes]; [ destruct g; discriminate Hm | ].
    cbn [nth_error] in Hm.
    refine (IH _ (wchain_next P) _ g k W p m (wchain_shift P g W HW) Hm Hpar
              Hp Hk).
    intros v (W0 & R & HW0 & HR & HvR & Hvv).
    destruct (field_sources_cases WS (Field_return_of_call (Normal 0)))
      as [[Hany _] | (S' & Hs' & Hnot' & Hiff')];
      [ rewrite Hany; exact I | rewrite Hs'; cbn [src_has] ].
    apply Hiff'. exists W0, R.
    split; [ exact (constructor_fact _ _ _ HR) | ].
    split; [ exact (HWS W0 HW0) | ].
    destruct HvR as (_ & HsR & _).
    split; [ | exact (proj1 Hvv) ].
    destruct HsR as [Has | Hs]; [ | exact Hs ].
    rewrite (Hnot' W0 R (constructor_fact _ _ _ HR) (HWS W0 HW0)) in Has.
    discriminate Has.
Qed.

(* What the sources [src] are of the node [B]: sources of it, or it has
   any source. *)
Definition src_covers (src : allocation_sources) (B : node) : Prop :=
  match src with
  | Any_source => as_ B
  | Sources Ss =>
    ∀ x : node, Ss x = true -> s_sources sol B x = true ∨ as_ B
  end.

Lemma call_witness_unknown_not_local :
  is_local (Field_call_witness Unknown_arity_code_pointer) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

Lemma result_not_local (i : nat) :
  is_local (Field_return_of_call (Normal i)) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

(* The witnesses of unknown arity that the witnesses of a call stand for
   are used ([call_reads] says it of the first). *)
Lemma call_reads_hu (P : node -> Prop) (ws returns : list node)
    (exn : node) :
  call_reads P ws returns exn ->
  ∀ (g : nat) (W w : node),
    wchain P g W -> nth_error ws g = Some w -> hu W.
Proof.
  intros (Hcover & _ & _ & Hnext & (called_node & Hau & Hcalled) & H0).
  induction g as [ | g IH]; intros W w HW Hw; [ exact (H0 W w HW Hw) | ].
  destruct HW as (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HWW).
  assert (Hwg : ∃ wg : node, nth_error ws g = Some wg).
  { destruct (nth_error ws g) as [wg | ] eqn:E; [ eauto | ].
    apply nth_error_None in E.
    assert (S g < length ws) by (apply nth_error_Some; congruence). lia. }
  destruct Hwg as (wg & Hwg).
  destruct (Hnext g wg w Hwg Hw) as (vg & Hvg & Hwv).
  pose proof (may_come_from_read _ _ _ Hc v W' wg vg R _ (result_not_local 0)
                HR Hvg (Hcover g W' wg HW' Hwg) HvR) as Hvvg.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ (au_hu _ _ _ Hc _ Hau)
                (Hcalled (S g) w Hw)) as Hhu_w.
  exact (read_hu v vg w W _ call_witness_unknown_not_local Hcons Hwv Hvvg
           Hhu_w).
Qed.

(* Code whose witness of unknown arity is used keeps its calling
   convention ([cc_code_id_unknown_arity]). *)
Lemma unchanged_of_unknown (id : code_id) (c : code) (y W : node) :
  g_constructor G y (Field_call_witness Unknown_arity_code_pointer) W ->
  hu W ->
  g_constructor G W Field_code_id_of_call_witness (Node_code_id id) ->
  unchanged id c.
Proof.
  intros Hcons Hhu Hcode.
  pose proof (cc_code_id_unknown_arity _ _ _ _ Hcc_closed y W _ Hcons Hhu
                Hcode) as Hcannot.
  unfold unchanged, code_calling_convention_change,
    cannot_change_calling_convention.
  rewrite Hcannot, orb_true_r. reflexivity.
Qed.

(* The arguments of a call of unknown arity whose first witness has any
   source have any usage. *)
Lemma caller_chain_any (returns : list node) (exn called_node : node) :
  ∀ L : list (list simple * node),
    caller_chain_edges G returns exn called_node L ->
    (∀ (grp : list simple) (c : node), hd_error L = Some (grp, c) -> as_ c)
    ->
    ∀ (g k : nat) (grp : list simple) (c : node) (s : simple),
      nth_error L g = Some (grp, c) -> nth_error grp k = Some s ->
      au (node_of_simple s).
Proof.
  induction L as [ | [grp0 c0] L IH]; intros Hch Has g k grp c s Hg Hs;
    [ destruct g; discriminate Hg | ].
  pose proof (Has grp0 c0 eq_refl) as Hc0.
  destruct Hch as (Hargs & _ & _ & Hnext & Hch).
  destruct g as [ | g].
  - simpl in Hg. injection Hg as <- <-.
    exact (any_usage_argument _ _ _ Hc _ _ _ Hc0 (Hargs k s Hs)).
  - simpl in Hg. refine (IH Hch _ g k grp c s Hg Hs).
    intros grp1 c1 Hhd.
    destruct L as [ | [grp1' c1'] L]; [ discriminate Hhd | ].
    simpl in Hhd. injection Hhd as <- <-.
    simpl in Hnext. destruct Hnext as (v & Hv & Hc1).
    exact (any_source_accessor _ _ _ Hc _ _ _
             (any_source_accessor _ _ _ Hc _ _ _ Hc0 Hv (result_not_local 0))
             Hc1 call_witness_unknown_not_local).
Qed.

(* An argument that the mask of a call of unknown arity keeps has a use:
   the mask keeps the arguments at the parameters that have a use of the
   witnesses that the sources of the callee lead to, group by group, and
   these are sources of the witnesses of the call. *)
Lemma unknown_mask_kept (returns : list node) (exn called_node : node) :
  ∀ (L : list (list simple * node)) (sizes : list nat)
    (src : allocation_sources) (B : node),
    caller_chain_edges G returns exn called_node L ->
    src_covers src B ->
    (∀ (grp : list simple) (c : node),
       hd_error L = Some (grp, c) ->
       ∃ a : node,
         g_accessor G a (Field_call_witness Unknown_arity_code_pointer) B
         ∧ (a = c ∨ s_alias sol c a = true)) ->
    ∀ (g k : nat) (grp : list simple) (c : node) (m : list bool)
      (s : simple),
      nth_error L g = Some (grp, c) ->
      nth_error
        (arguments_used_by_call F sol Unknown_arity_code_pointer src sizes) g
      = Some m ->
      nth_error grp k = Some s -> nth_error m k = Some true ->
      hu (node_of_simple s).
Proof.
  induction L as [ | [grp0 c0] L IH];
    intros sizes src B Hch Hcov Hread g k grp c m s Hg Hm Hs Hk;
    [ destruct g; discriminate Hg | ].
  destruct (Hread grp0 c0 eq_refl) as (a & Ha & Hac).
  (* Any source at [c0] gives every argument any usage. *)
  assert (Hany : as_ c0 -> hu (node_of_simple s)).
  { intros Hc0. apply (au_hu _ _ _ Hc).
    refine (caller_chain_any returns exn called_node _ Hch _ g k grp c s Hg
              Hs).
    intros grp' c' Hhd. simpl in Hhd. injection Hhd as _ <-. exact Hc0. }
  assert (Hac0 : as_ a -> as_ c0).
  { intros Ea. destruct Hac as [<- | Hal]; [ exact Ea | ].
    exact (any_source_alias _ _ _ Hc _ _ Hal Ea). }
  assert (HB : as_ B -> as_ c0).
  { intros EB. apply Hac0.
    exact (any_source_accessor _ _ _ Hc _ _ _ EB Ha
             call_witness_unknown_not_local). }
  destruct sizes as [ | n sizes]; [ destruct g; discriminate Hm | ].
  cbn [arguments_used_by_call] in Hm.
  destruct src as [ | Ss]; [ exact (Hany (HB Hcov)) | ].
  destruct (field_sources_cases Ss
              (Field_call_witness Unknown_arity_code_pointer))
    as [[Hfs (x & y & Hxy & Hx & Ey)] | (WS & Hws & Hnot & Hiff)].
  { apply Hany.
    destruct (Hcov x Hx) as [Hsx | EB]; [ | exact (HB EB) ].
    destruct (s_any_source sol B) eqn:EB; [ exact (HB eq_refl) | ].
    apply Hac0.
    exact (any_source_alias _ _ _ Hc _ _
             (alias_accessor_source _ _ _ Hc _ _ _ _ _ Ha Hsx EB
                (fact_constructor _ _ _ Hxy)) Ey). }
  rewrite Hws in Hm.
  (* The witnesses of the first group are sources of [c0]. *)
  assert (HWS : ∀ z : node,
            WS z = true -> s_sources sol c0 z = true ∨ as_ c0).
  { intros z Hz. apply Hiff in Hz as (x & y & Hxy & Hx & Hyz & Hhz).
    destruct (s_any_source sol c0) eqn:Ec0; [ right; reflexivity | left ].
    destruct (Hcov x Hx) as [Hsx | EB]; [ | discriminate (HB EB) ].
    destruct (s_any_source sol B) eqn:EB; [ discriminate (HB eq_refl) | ].
    pose proof (alias_accessor_source _ _ _ Hc _ _ _ _ _ Ha Hsx EB
                  (fact_constructor _ _ _ Hxy)) as Hay.
    destruct (s_any_source sol a) eqn:Ea; [ discriminate (Hac0 eq_refl) | ].
    destruct (s_any_source sol y) eqn:Ey;
      [ rewrite (any_source_alias _ _ _ Hc _ _ Hay Ey) in Ea;
        discriminate Ea | ].
    pose proof (sources_alias _ _ _ Hc _ _ _ Hyz Ey Hay Ea) as Haz.
    destruct Hac as [<- | Hal]; [ exact Haz | ].
    exact (sources_alias _ _ _ Hc _ _ _ Haz Ea Hal Ec0). }
  destruct g as [ | g].
  - cbn [nth_error] in Hm. injection Hm as <-.
    simpl in Hg. injection Hg as <- <-.
    destruct (nth_error (seq 0 n) k) as [j | ] eqn:Hj;
      [ | rewrite nth_error_map, Hj in Hk; discriminate Hk ].
    rewrite nth_error_map, Hj in Hk. injection Hk as Hk.
    rewrite (nth_error_seq_eq _ _ _ _ Hj) in Hk. simpl in Hk.
    unfold cofield_has_use in Hk.
    apply existsb_exists in Hk as ([[z j'] p] & Hzp & Hb).
    apply andb_true_iff in Hb as [Hb Hp].
    apply andb_true_iff in Hb as [Hz Hji].
    apply Nat.eqb_eq in Hji. subst j'.
    destruct (HWS z Hz) as [Hsz | Ec0]; [ | exact (Hany Ec0) ].
    destruct (s_any_source sol c0) eqn:Ec0; [ exact (Hany eq_refl) | ].
    destruct Hch as (Hargs & _).
    exact (has_usage_alias _ _ _ Hc _ _ Hp
             (alias_argument_source _ _ _ Hc _ _ _ _ _ (Hargs k s Hs) Hsz
                Ec0 (fact_parameter _ _ _ Hzp))).
  - destruct sizes as [ | n' sizes]; [ destruct g; discriminate Hm | ].
    cbn [nth_error] in Hm. simpl in Hg.
    destruct L as [ | [grp1 c1] L]; [ destruct g; discriminate Hg | ].
    destruct Hch as (_ & _ & _ & Hnext & Hch). simpl in Hnext.
    destruct Hnext as (v & Hv & Hc1).
    refine (IH (n' :: sizes) _ v Hch _ _ g k grp c m s Hg Hm Hs Hk).
    + destruct (field_sources_cases WS (Field_return_of_call (Normal 0)))
        as [[Hfs (x & y & Hxy & Hx & Ey)] | (S' & Hs' & Hnot' & Hiff')];
        [ rewrite Hfs | rewrite Hs' ]; cbn [src_covers].
      * destruct (HWS x Hx) as [Hsx | Ec0];
          [ | exact (any_source_accessor _ _ _ Hc _ _ _ Ec0 Hv
                       (result_not_local 0)) ].
        destruct (s_any_source sol c0) eqn:Ec0;
          [ exact (any_source_accessor _ _ _ Hc _ _ _ Ec0 Hv
                     (result_not_local 0)) | ].
        exact (any_source_alias _ _ _ Hc _ _
                 (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hv Hsx Ec0
                    (fact_constructor _ _ _ Hxy)) Ey).
      * intros z Hz. apply Hiff' in Hz as (x & y & Hxy & Hx & Hyz & Hhz).
        destruct (s_any_source sol v) eqn:Ev; [ right; reflexivity | left ].
        destruct (HWS x Hx) as [Hsx | Ec0];
          [ | rewrite (any_source_accessor _ _ _ Hc _ _ _ Ec0 Hv
                         (result_not_local 0)) in Ev;
              discriminate Ev ].
        destruct (s_any_source sol c0) eqn:Ec0;
          [ rewrite (any_source_accessor _ _ _ Hc _ _ _ Ec0 Hv
                       (result_not_local 0)) in Ev;
            discriminate Ev | ].
        pose proof (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hv Hsx Ec0
                      (fact_constructor _ _ _ Hxy)) as Hvy.
        destruct (s_any_source sol y) eqn:Ey;
          [ rewrite (any_source_alias _ _ _ Hc _ _ Hvy Ey) in Ev;
            discriminate Ev | ].
        exact (sources_alias _ _ _ Hc _ _ _ Hyz Ey Hvy Ev).
    + intros grp' c' Hhd. simpl in Hhd. injection Hhd as <- <-.
      exists c1. split; [ exact Hc1 | left; reflexivity ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Arguments at the witnesses of a chain                                    *)
(* ------------------------------------------------------------------------ *)

Lemma args_rel_nth (P : node -> Prop) (a : arity) (vs vs' : list value) :
  args_rel P a vs vs' ->
  ∀ (i : nat) (v v' : value) (g k : nat) (W p : node),
    nth_error vs i = Some v -> nth_error vs' i = Some v' ->
    group_position (arity_sizes a) i = Some (g, k) ->
    wchain P g W -> g_parameter G W k p -> hu p -> rel v p v'.
Proof.
  intros [_ Hf] i v v' g k W p Hv Hv' Hpos HW Hp Hhu.
  destruct (fields_related_nth _ 0 _ _ i v Hf Hv) as (v1 & Hv1 & HP).
  rewrite Hv' in Hv1. injection Hv1 as <-.
  exact (HP g k W p Hpos HW Hp Hhu).
Qed.

Lemma args_rel_intro (P : node -> Prop) (a : arity) (vs vs' : list value) :
  map kind_of_value vs' = map kind_of_value vs ->
  (∀ (i : nat) (v v' : value) (g k : nat) (W p : node),
     nth_error vs i = Some v -> nth_error vs' i = Some v' ->
     group_position (arity_sizes a) i = Some (g, k) ->
     wchain P g W -> g_parameter G W k p -> hu p -> rel v p v') ->
  args_rel P a vs vs'.
Proof.
  intros Hk H. split; [ exact Hk | ].
  apply fields_related_intro.
  - rewrite <- (length_map kind_of_value vs'), Hk, length_map. reflexivity.
  - intros i v v' Hv Hv' g k W p Hpos HW Hp Hhu.
    exact (H i v v' g k W p Hv Hv' Hpos HW Hp Hhu).
Qed.

Lemma args_rel_length (P : node -> Prop) (a : arity) (vs vs' : list value) :
  args_rel P a vs vs' -> length vs' = length vs.
Proof.
  intros [Hk _]. rewrite <- (length_map kind_of_value vs'), Hk, length_map.
  reflexivity.
Qed.

(* The arguments that a partial application keeps, then those of the
   call of it. *)
Lemma args_rel_app (PM PN : node -> Prop) (aa a : arity)
    (applied applied' vs vs' : list value) :
  args_rel PM aa applied applied' -> args_rel PN a vs vs' ->
  (∀ (g : nat) (W : node), wchain PM (length aa + g) W -> wchain PN g W) ->
  length applied = length (unarize aa) ->
  args_rel PM (aa ++ a) (applied ++ vs) (applied' ++ vs').
Proof.
  intros Haa Ha Hshift Hlen.
  pose proof (args_rel_length _ _ _ _ Haa) as Hlen'.
  apply args_rel_intro.
  - rewrite !map_app, (proj1 Haa), (proj1 Ha). reflexivity.
  - intros i v v' g k W p Hv Hv' Hpos HW Hp Hhu.
    rewrite arity_sizes_app in Hpos.
    assert (Hsum : list_sum (arity_sizes aa) = length applied)
      by (rewrite sum_arity_sizes; symmetry; exact Hlen).
    destruct (Nat.ltb_spec i (length applied)) as [Hi | Hi].
    + rewrite nth_error_app1 in Hv by exact Hi.
      rewrite nth_error_app1 in Hv' by lia.
      rewrite group_position_app_l in Hpos by lia.
      exact (args_rel_nth _ _ _ _ Haa i v v' g k W p Hv Hv' Hpos HW Hp Hhu).
    + rewrite nth_error_app2 in Hv by exact Hi.
      rewrite nth_error_app2 in Hv' by lia. rewrite Hlen' in Hv'.
      rewrite group_position_app_r in Hpos by lia.
      rewrite Hsum in Hpos.
      destruct (group_position (arity_sizes a) (i - length applied))
        as [[g' k'] | ] eqn:E; [ | discriminate Hpos ].
      injection Hpos as <- <-.
      rewrite length_arity_sizes in HW.
      exact (args_rel_nth _ _ _ _ Ha _ v v' g' k' W p Hv Hv' E
               (Hshift g' W HW) Hp Hhu).
Qed.

Lemma map_skipn_comm {A B : Type} (f : A -> B) (n : nat) (l : list A) :
  map f (skipn n l) = skipn n (map f l).
Proof.
  revert l. induction n as [ | n IH]; intros [ | x l]; simpl;
    [ reflexivity | reflexivity | reflexivity | apply IH ].
Qed.

(* The arguments left after the groups of the code. *)
Lemma args_rel_skipn (PM PR : node -> Prop) (A : arity)
    (all all' : list value) (p : nat) :
  args_rel PM A all all' ->
  (∀ (g : nat) (W : node), wchain PR g W -> wchain PM (p + g) W) ->
  p <= length A ->
  args_rel PR (skipn p A) (skipn (list_sum (firstn p (arity_sizes A))) all)
    (skipn (list_sum (firstn p (arity_sizes A))) all').
Proof.
  intros HA Hshift Hp.
  set (off := list_sum (firstn p (arity_sizes A))).
  apply args_rel_intro.
  - rewrite !map_skipn_comm, (proj1 HA). reflexivity.
  - intros i v v' g k W q Hv Hv' Hpos HW Hq Hhu.
    rewrite nth_error_skipn_add in Hv, Hv'.
    refine (args_rel_nth _ _ _ _ HA (off + i) v v' (p + g) k W q Hv Hv' _
              (Hshift g W HW) Hq Hhu).
    rewrite <- (firstn_skipn p (arity_sizes A)).
    rewrite group_position_app_r by (unfold off; lia).
    replace (off + i - list_sum (firstn p (arity_sizes A))) with i
      by (unfold off; lia).
    rewrite <- arity_sizes_skipn, Hpos.
    rewrite length_firstn, length_arity_sizes, Nat.min_l by exact Hp.
    reflexivity.
Qed.

(* From what may come to the first result of a witness of the chain, the
   chain goes on. *)
Lemma wchain_from_return (P : node -> Prop) (g0 : nat) (W0 R0 : node) :
  wchain P g0 W0 ->
  g_constructor G W0 (Field_return_of_call (Normal 0)) R0 ->
  ∀ (g : nat) (W : node), wchain (from_node R0) g W -> wchain P (S g0 + g) W.
Proof.
  intros HW0 HR0. induction g as [ | g IH]; intros W HW.
  - rewrite Nat.add_0_r. destruct HW as (y & [HyR Hyy] & Hcons & HWW).
    exists W0, R0, y. auto 10.
  - destruct HW as (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HWW).
    rewrite Nat.add_succ_r. exists W', R, v.
    split; [ exact (IH W' HW') | auto 10 ].
Qed.

(* What a partial application at [N] says of the first results of the
   last witnesses of its chain: the chain goes on from [N]. *)
Lemma wchain_shift_end (PM : node -> Prop) (N : node) (l : nat) :
  (∀ (g0 : nat) (W R y : node),
     S g0 = l -> wchain PM g0 W ->
     g_constructor G W (Field_return_of_call (Normal 0)) R ->
     may_come_from sol y R -> may_come_from sol y y ->
     may_come_from sol y N) ->
  l <> 0 ->
  ∀ (g : nat) (W : node), wchain PM (l + g) W -> wchain (from_node N) g W.
Proof.
  intros Hend Hl. induction g as [ | g IH]; intros W HW.
  - destruct l as [ | g0]; [ contradiction | ]. rewrite Nat.add_0_r in HW.
    destruct HW as (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HWW).
    exists v.
    split; [ split; [ exact (Hend g0 W' R v eq_refl HW' HR HvR Hvv)
                    | exact Hvv ]
           | auto 10 ].
  - rewrite Nat.add_succ_r in HW.
    destruct HW as (W' & R & v & HW' & HR & HvR & Hvv & Hcons & HWW).
    exists W', R, v. split; [ exact (IH W' HW') | auto 10 ].
Qed.

(* A closure related is observed equal. *)
Lemma rel_closure_observed (statics : static_env) (v v' : value) (n : node)
    (id : code_id) (slots : list (value_slot * value)) :
  rel v n v' -> closure_of statics v = Some (id, slots) -> observed_equal v v'.
Proof.
  intros Hrel Hcl.
  destruct v as [c | code vs | slot decls vs | s | tag fields | elements | c
                | l | callee args arity];
    simpl in Hcl; try discriminate Hcl; cbn [rel] in Hrel.
  - destruct Hrel.
  - destruct Hrel as (sites & vs' & _ & -> & _). cbn [observed_equal]. auto.
  - destruct Hrel as (-> & _). apply observed_equal_refl.
Qed.

(* An indirect call of the fragment whose callee is not a partial
   application, in a unit whose code is not tupled and has as many
   parameters as the call has arguments: it goes wrong, or runs the body
   of the code of its callee, which the unit binds. *)
Lemma indirect_call_step (ap : apply_expr) (vars : env) (K : cont_env)
    (statics : static_env) (hp : heap) (rcd : read_record) :
  ap_call_kind ap = Indirect ->
  (∀ (id : code_id) (c : code),
     (id, c) ∈ codes ->
     code_is_tupled c = false
     ∧ length (code_params_arity c) = length (ap_args_arity ap)) ->
  codes_ok is_local G noted deps cds keep codes symbols False statics ->
  (∀ callee : value,
     eval_simple vars (ap_callee ap) = Some callee ->
     partial_application_of callee = None) ->
  match step (State (Apply ap) vars K statics hp rcd) with
  | Step s' =>
    ∃ (callee : value) (vs extra : list value) (id : code_id)
      (slots : list (value_slot * value)) (c : code) (callee_vars : env),
      eval_simple vars (ap_callee ap) = Some callee
      ∧ eval_simples vars (ap_args ap) = Some vs
      ∧ eval_simples vars (exn_extra_simples ap) = Some extra
      ∧ closure_of statics callee = Some (id, slots)
      ∧ lookup id (static_code statics) = Some c
      ∧ kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)) = true
      ∧ kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
          (unarize_per_parameter (code_params_arity c)) = true
      ∧ arity_eqb (ap_return_arity ap) (code_result_arity c) = true
      ∧ bind_params (code_params c) vs [(code_my_closure c, callee)]
        = Some callee_vars
      ∧ s' = State (code_body c) callee_vars
               [(code_return_continuation c, resolve (ap_continuation ap) K);
                (code_exn_continuation c,
                 with_extra_args (resolve (ap_exn_continuation ap) K) extra)]
               statics hp rcd
  | Exit _ _ _ _ => False
  | Undefined => True
  end.
Proof.
  intros Hkind Hfull Hcodes Hnpa. rewrite step_apply.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs; [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
    eqn:Hextra; [ | exact I ].
  unfold call_function. rewrite Hkind.
  assert (Hshape :
    shape_of_call statics callee Indirect vs (ap_args_arity ap)
      (ap_return_arity ap)
    = shape_of_closure_call statics callee Indirect vs (ap_args_arity ap)
        (ap_return_arity ap)).
  { unfold shape_of_call. rewrite (Hnpa callee eq_refl). reflexivity. }
  rewrite Hshape.
  destruct (closure_of statics callee) as [[id slots] | ] eqn:Hclosure;
    [ | unfold shape_of_closure_call; rewrite Hclosure; exact I ].
  destruct (lookup id (static_code statics)) as [c | ] eqn:Hcode;
    [ | unfold shape_of_closure_call; rewrite Hclosure, Hcode; exact I ].
  destruct (Hcodes id c Hcode) as [Hin _].
  destruct (Hfull id c Hin) as [Htupled Hlen].
  rewrite (shape_of_closure_call_full statics callee Indirect vs
             (ap_args_arity ap) (ap_return_arity ap) id slots c Hclosure
             (or_introl eq_refl) Hcode Htupled (eq_sym Hlen)).
  simpl. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)))
    eqn:Hk1; [ | exact I ].
  destruct (kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
              (unarize_per_parameter (code_params_arity c))) eqn:Hk2;
    [ | exact I ].
  destruct (arity_eqb (ap_return_arity ap) (code_result_arity c)) eqn:Hk3;
    [ | exact I ].
  simpl.
  destruct (bind_params (code_params c) vs [(code_my_closure c, callee)])
    as [callee_vars | ] eqn:Hbind; [ | exact I ].
  exists callee, vs, extra, id, slots, c, callee_vars. auto 12.
Qed.

Lemma lookup_rets_none (rets : finite_map (list flambda_kind))
    (k : continuation) :
  (∀ K : list flambda_kind, lookup k rets = Some K -> False) ->
  lookup k rets = None.
Proof.
  intros H. destruct (lookup k rets) eqn:E; [ | reflexivity ].
  exfalso. exact (H _ eq_refl).
Qed.

(* The call of [make_apply_wrapper] in the rebuild of a call. *)
Definition rebuilt_call (en : rebuild_env) (ap : apply_expr) : apply_expr :=
  {| ap_callee := ap_callee ap;
     ap_call_kind :=
       rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap;
     ap_args := rebuild_arguments F sol en ap;
     ap_continuation := ap_continuation ap;
     ap_exn_continuation := ap_exn_continuation ap;
     ap_exn_extra_args := rebuild_exn_extra_args sol en ap;
     ap_args_arity := ap_args_arity ap;
     ap_return_arity := ap_return_arity ap |}.

(* What [answered] asks of the second state [t] where the first [s]
   steps to [s']. *)
Definition answered_step (s s' t : state) : Prop :=
  step s' = Undefined
  ∨ (∃ t' : state, step t = Step t' ∧ related s' t')
  ∨ (measure s' < measure s ∧ related s' t)
  ∨ (∃ (t' : state) (n : nat), run (S (S n)) t = Step t' ∧ related s' t').

(* Entering the body of code that is used, with the parameters and
   [my_closure] related, and the continuations of the code ways out at
   its nodes: the second run enters the rebuilt body, in a related
   state. *)
Lemma code_entered (id : code_id) (params : list variable) (mc : variable)
    (ret exn_k : continuation) (md : code_metadata) (cbody : expr)
    (cd : code_dep) (callee : value) (vs : list value)
    (callee_vars callee_vars' : env) (rt rt' et et' : cont_value)
    (statics statics' : static_env) (hp : heap) (rcd rcd' : read_record) :
  unchanged id (Code params mc ret exn_k md cbody) ->
  flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc
    (State cbody callee_vars [(ret, rt); (exn_k, et)] statics hp rcd) ->
  statics_rel statics statics' ->
  scrutinees_used G (Some id) cbody ->
  checked exact codes cbody = true ->
  raises_kept [] cbody = true ->
  code_is_closed (Code params mc ret exn_k md cbody) = true ->
  calls_kinded [] cbody = true ->
  among first_fresh (continuations_of cbody) = false ->
  ret <> exn_k ->
  conts_counted [(exn_k, 1); (ret, length (unarize (cm_result_arity md)))]
    cbody = true ->
  edges (Some id) [(exn_k, [cd_exn cd]); (ret, cd_return cd)] cbody ->
  length (cd_return cd) = length (unarize (cm_result_arity md)) ->
  map kind_of_value vs = unarize (cm_params_arity md) ->
  bind_params params vs [(mc, callee)] = Some callee_vars ->
  env_ok (rev (combine params (map kind_of_value vs)) ++ [(mc, Value)])
    callee_vars callee_vars' ->
  au (Node_code_id id) ->
  way_out cont_rel over_rel (cd_return cd) rt rt' ->
  way_out cont_rel over_rel [cd_exn cd] et et' ->
  related (State cbody callee_vars [(ret, rt); (exn_k, et)] statics hp rcd)
    (State (code_body (rebuild_c id (Code params mc ret exn_k md cbody)))
       callee_vars' [(ret, rt'); (exn_k, et')] statics' hp rcd').
Proof.
  intros Hun Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
    Hne Hcounted Hbody Lr Hkp Hbind Hok' Hlive' Hway_ret Hway_exn'.
  assert (Hheap : hp = hp) by reflexivity.
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
    as (Hcvars & _).
  simpl_state.
  set (en' := bind_variables FlambdaReaperRebuild.empty_env
                ((mc, Value)
                 :: with_kinds params (unarize (cm_params_arity md)))).
  split; [ exact Hflow' | ].
  exists (Some id), en', (code_conts (Code params mc ret exn_k md cbody) cd),
    [].
  simpl_state.
  rewrite (rebuilt_code_body id (Code params mc ret exn_k md cbody) callee vs
             callee_vars Hun Hbind Hcvars).
  fold en'.
  split; [ exact Hbody | split; [ reflexivity | ] ].
  split.
  { unfold en'.
    cbn [env_kinds bind_variables bind_variables_then
         FlambdaReaperRebuild.empty_env rev].
    rewrite app_nil_r.
    rewrite (with_kinds_combine params); [ rewrite <- Hkp; exact Hok' | ].
    rewrite <- Hkp, length_map.
    exact (bind_params_length _ _ _ _ Hbind). }
  split.
  { (* The continuations of the code are the ways out of the call. *)
    assert (Hfree_c : ∀ k : continuation,
              k ∈ free_continuations_of_expr cbody -> k = ret ∨ k = exn_k).
    { intros k Hk. cbn [code_is_closed] in Hclosed_c.
      apply andb_true_iff in Hclosed_c as [Hcl _].
      apply andb_true_iff in Hcl as [_ Hcl].
      cbn [free_continuations_of_code] in Hcl.
      destruct (among k [ret; exn_k]) eqn:Hkin.
      - apply among_In in Hkin. destruct Hkin as [<- | [<- | []]]; auto.
      - exfalso.
        assert (Hin : k ∈ without [ret; exn_k]
                            (free_continuations_of_expr cbody)).
        { unfold without. apply filter_In. split; [ exact Hk | ].
          rewrite Hkin. reflexivity. }
        destruct (without [ret; exn_k] (free_continuations_of_expr cbody));
          [ destruct Hin | discriminate Hcl ]. }
    intros k Hk. left. split; [ reflexivity | ].
    pose proof (proj1 free_rebuild cbody Hcf_c en' k Hk) as Hk0.
    unfold code_conts. cbn [code_exn_continuation code_return_continuation].
    destruct (Hfree_c k Hk0) as [-> | ->].
    - exists (cd_return cd). unfold resolve. cbn [lookup].
      rewrite Pos.eqb_refl.
      destruct (Pos.eqb_spec ret exn_k) as [E | _]; [ contradiction | ].
      split; [ reflexivity | exact Hway_ret ].
    - exists [cd_exn cd]. unfold resolve. cbn [lookup].
      rewrite Pos.eqb_refl.
      destruct (Pos.eqb_spec exn_k ret) as [E | _];
        [ symmetry in E; contradiction | ].
      split; [ reflexivity | exact Hway_exn' ]. }
  split; [ exact Hst | split; [ exact Hheap | ] ].
  split; [ | left; reflexivity ].
  split; [ exact Hlive' | ].
  split; [ | split; [ exact Hscr_c | split; [ exact Hcf_c | ] ] ].
  - exists []. split; [ split; [ intros k Hk; discriminate Hk
                              | intros r K Hr; discriminate Hr ] | ].
    apply (proj1 exits_bound_of_counted cbody _ [] _ Hcounted).
    intros k m Hl _. unfold code_conts.
    cbn [code_exn_continuation code_return_continuation lookup] in Hl |- *.
    destruct (Pos.eqb k exn_k);
      [ injection Hl as <-; exists [cd_exn cd]; split; reflexivity | ].
    destruct (Pos.eqb k ret); [ | discriminate Hl ].
    injection Hl as <-. exists (cd_return cd).
    split; [ reflexivity | exact Lr ].
  - split; [ | intros r K Hr; discriminate Hr ].
    split; [ exact Hfresh_c | ].
    split.
    + exists []. split; [ exact Hkinded_c | ].
      intros k m Hl _. unfold en' in Hl. cbn in Hl. discriminate Hl.
    + exists []. split; [ exact Hraises_c | ].
      intros k. unfold en'. cbn. exact I.
Qed.

(* Entering the body of code that changes its calling convention: as
   [code_entered], the return continuation of the code being that of a
   code that changes ([ret_rel]), with the mask of the results kept. *)
Lemma code_entered_cc (id : code_id) (params : list variable) (mc : variable)
    (ret exn_k : continuation) (md : code_metadata) (cbody : expr)
    (cd : code_dep) (pd rd : list param_decision) (callee : value)
    (vs : list value)
    (callee_vars callee_vars' : env) (rt rt' et et' : cont_value)
    (statics statics' : static_env) (hp : heap) (rcd rcd' : read_record) :
  changing id (Code params mc ret exn_k md cbody) pd rd ->
  flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc
    (State cbody callee_vars [(ret, rt); (exn_k, et)] statics hp rcd) ->
  statics_rel statics statics' ->
  scrutinees_used G (Some id) cbody ->
  checked exact codes cbody = true ->
  raises_kept [] cbody = true ->
  code_is_closed (Code params mc ret exn_k md cbody) = true ->
  calls_kinded [] cbody = true ->
  among first_fresh (continuations_of cbody) = false ->
  no_raise_to ret cbody = true ->
  rcalls_ok (rev (with_kinds params (unarize (cm_params_arity md)))
             ++ [(mc, Value)])
    ret (unarize (cm_result_arity md)) cbody = true ->
  ret <> exn_k ->
  conts_counted [(exn_k, 1); (ret, length (unarize (cm_result_arity md)))]
    cbody = true ->
  edges (Some id) [(exn_k, [cd_exn cd]); (ret, cd_return cd)] cbody ->
  length (cd_return cd) = length (unarize (cm_result_arity md)) ->
  map kind_of_value vs = unarize (cm_params_arity md) ->
  bind_params params vs [(mc, callee)] = Some callee_vars ->
  env_ok (rev (combine params (map kind_of_value vs)) ++ [(mc, Value)])
    callee_vars callee_vars' ->
  au (Node_code_id id) ->
  length rd = length (unarize (cm_result_arity md)) ->
  ret_rel cont_rel over_rel (map is_kept rd) (unarize (cm_result_arity md))
    (cd_return cd) rt rt' ->
  way_out cont_rel over_rel [cd_exn cd] et et' ->
  related (State cbody callee_vars [(ret, rt); (exn_k, et)] statics hp rcd)
    (State (code_body (rebuild_c id (Code params mc ret exn_k md cbody)))
       callee_vars' [(ret, rt'); (exn_k, et')] statics' hp rcd').
Proof.
  intros Hch Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
    Hraise Hrc Hne Hcounted Hbody Lr Hkp Hbind Hok' Hlive' Lrd Hret
    Hway_exn'.
  assert (Hheap : hp = hp) by reflexivity.
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
    as (Hcvars & _).
  simpl_state.
  set (en' := bind_variables
                (bind_continuations FlambdaReaperRebuild.empty_env
                   [(ret, map is_kept rd)])
                ((mc, Value)
                 :: with_kinds params (unarize (cm_params_arity md)))).
  split; [ exact Hflow' | ].
  exists (Some id), en', (code_conts (Code params mc ret exn_k md cbody) cd),
    [(ret, unarize (cm_result_arity md))].
  simpl_state.
  rewrite (rebuilt_code_body_cc id (Code params mc ret exn_k md cbody) pd rd
             callee vs callee_vars Hch Hbind Hcvars).
  fold en'.
  split; [ exact Hbody | split; [ reflexivity | ] ].
  split.
  { unfold en'.
    cbn [env_kinds bind_variables bind_variables_then bind_continuations
         FlambdaReaperRebuild.empty_env rev].
    rewrite app_nil_r.
    rewrite (with_kinds_combine params); [ rewrite <- Hkp; exact Hok' | ].
    rewrite <- Hkp, length_map.
    exact (bind_params_length _ _ _ _ Hbind). }
  split.
  { (* The continuations of the code are the ways out of the call. *)
    assert (Hfree_c : ∀ k : continuation,
              k ∈ free_continuations_of_expr cbody -> k = ret ∨ k = exn_k).
    { intros k Hk. cbn [code_is_closed] in Hclosed_c.
      apply andb_true_iff in Hclosed_c as [Hcl _].
      apply andb_true_iff in Hcl as [_ Hcl].
      cbn [free_continuations_of_code] in Hcl.
      destruct (among k [ret; exn_k]) eqn:Hkin.
      - apply among_In in Hkin. destruct Hkin as [<- | [<- | []]]; auto.
      - exfalso.
        assert (Hin : k ∈ without [ret; exn_k]
                            (free_continuations_of_expr cbody)).
        { unfold without. apply filter_In. split; [ exact Hk | ].
          rewrite Hkin. reflexivity. }
        destruct (without [ret; exn_k] (free_continuations_of_expr cbody));
          [ destruct Hin | discriminate Hcl ]. }
    intros k Hk.
    pose proof (proj1 free_rebuild cbody Hcf_c en' k Hk) as Hk0.
    unfold code_conts, en'.
    cbn [code_exn_continuation code_return_continuation env_conts
         bind_variables bind_variables_then bind_continuations
         FlambdaReaperRebuild.empty_env app].
    destruct (Hfree_c k Hk0) as [-> | ->].
    - right. right. exists (unarize (cm_result_arity md)).
      split; [ cbn [lookup]; rewrite Pos.eqb_refl; reflexivity | ].
      exists (map is_kept rd), (cd_return cd). unfold resolve. cbn [lookup].
      rewrite Pos.eqb_refl.
      destruct (Pos.eqb_spec ret exn_k) as [E | _]; [ contradiction | ].
      split; [ left; reflexivity | ].
      split; [ rewrite length_map; exact Lrd | ].
      split; [ reflexivity | exact Hret ].
    - left. cbn [lookup].
      destruct (Pos.eqb_spec exn_k ret) as [E | _];
        [ symmetry in E; contradiction | ].
      split; [ reflexivity | ].
      exists [cd_exn cd]. unfold resolve. cbn [lookup].
      rewrite Pos.eqb_refl.
      destruct (Pos.eqb_spec exn_k ret) as [E | _];
        [ symmetry in E; contradiction | ].
      split; [ reflexivity | exact Hway_exn' ]. }
  split; [ exact Hst | split; [ exact Hheap | ] ].
  split; [ | left; reflexivity ].
  split; [ exact Hlive' | ].
  split; [ | split; [ exact Hscr_c | split; [ exact Hcf_c | ] ] ].
  - exists []. split; [ split; [ intros k Hk; discriminate Hk
                              | intros r K _; reflexivity ] | ].
    apply (proj1 exits_bound_of_counted cbody _ [] _ Hcounted).
    intros k m Hl _. unfold code_conts.
    cbn [code_exn_continuation code_return_continuation lookup] in Hl |- *.
    destruct (Pos.eqb k exn_k);
      [ injection Hl as <-; exists [cd_exn cd]; split; reflexivity | ].
    destruct (Pos.eqb k ret); [ | discriminate Hl ].
    injection Hl as <-. exists (cd_return cd).
    split; [ reflexivity | exact Lr ].
  - split.
    2: { intros r K Hr. cbn [lookup] in Hr.
         destruct (Pos.eqb_spec r ret) as [-> | _]; [ | discriminate Hr ].
         injection Hr as <-. split; [ exact Hraise | ].
         unfold en'.
         cbn [env_kinds bind_variables bind_variables_then bind_continuations
              FlambdaReaperRebuild.empty_env rev].
         rewrite app_nil_r. exact Hrc. }
    split; [ exact Hfresh_c | ].
    split.
    + exists [].
      split; [ exact Hkinded_c | ].
      intros k m Hl Hkr. unfold en' in Hl. cbn in Hl.
      destruct (Pos.eqb_spec k ret) as [-> | _]; [ | discriminate Hl ].
      cbn [lookup] in Hkr. rewrite Pos.eqb_refl in Hkr. discriminate Hkr.
    + exists [(ret, false)].
      split; [ exact (proj1 (raises_kept_unraised ret false) cbody []
                        Hraise Hraises_c) | ].
      intros k. unfold en'. cbn.
      destruct (Pos.eqb k ret); [ intros E; discriminate E | exact I ].
Qed.

(* Entering the body of code that keeps its calling convention, whose
   return continuation is that of code that changes ([ret_rel]), every
   result kept: as [code_entered_cc]. *)
Lemma code_entered_ret (id : code_id) (params : list variable) (mc : variable)
    (ret exn_k : continuation) (md : code_metadata) (cbody : expr)
    (cd : code_dep) (callee : value)
    (vs : list value)
    (callee_vars callee_vars' : env) (rt rt' et et' : cont_value)
    (statics statics' : static_env) (hp : heap) (rcd rcd' : read_record) :
  unchanged id (Code params mc ret exn_k md cbody) ->
  flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc
    (State cbody callee_vars [(ret, rt); (exn_k, et)] statics hp rcd) ->
  statics_rel statics statics' ->
  scrutinees_used G (Some id) cbody ->
  checked exact codes cbody = true ->
  raises_kept [] cbody = true ->
  code_is_closed (Code params mc ret exn_k md cbody) = true ->
  calls_kinded [] cbody = true ->
  among first_fresh (continuations_of cbody) = false ->
  no_raise_to ret cbody = true ->
  (length params = length (unarize (cm_params_arity md)) ->
   rcalls_ok (rev (with_kinds params (unarize (cm_params_arity md)))
              ++ [(mc, Value)])
     ret (unarize (cm_result_arity md)) cbody = true) ->
  ret <> exn_k ->
  conts_counted [(exn_k, 1); (ret, length (unarize (cm_result_arity md)))]
    cbody = true ->
  edges (Some id) [(exn_k, [cd_exn cd]); (ret, cd_return cd)] cbody ->
  length (cd_return cd) = length (unarize (cm_result_arity md)) ->
  map kind_of_value vs = unarize (cm_params_arity md) ->
  bind_params params vs [(mc, callee)] = Some callee_vars ->
  env_ok (rev (combine params (map kind_of_value vs)) ++ [(mc, Value)])
    callee_vars callee_vars' ->
  au (Node_code_id id) ->
  ret_rel cont_rel over_rel
    (repeat true (length (unarize (cm_result_arity md))))
    (unarize (cm_result_arity md)) (cd_return cd) rt rt' ->
  way_out cont_rel over_rel [cd_exn cd] et et' ->
  related (State cbody callee_vars [(ret, rt); (exn_k, et)] statics hp rcd)
    (State (code_body (rebuild_c id (Code params mc ret exn_k md cbody)))
       callee_vars' [(ret, rt'); (exn_k, et')] statics' hp rcd').
Proof.
  intros Hun Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
    Hraise Hrc Hne Hcounted Hbody Lr Hkp Hbind Hok' Hlive' Hret
    Hway_exn'.
  assert (Hheap : hp = hp) by reflexivity.
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
    as (Hcvars & _).
  simpl_state.
  set (en' := bind_variables FlambdaReaperRebuild.empty_env
                ((mc, Value)
                 :: with_kinds params (unarize (cm_params_arity md)))).
  split; [ exact Hflow' | ].
  exists (Some id), en', (code_conts (Code params mc ret exn_k md cbody) cd),
    [(ret, unarize (cm_result_arity md))].
  simpl_state.
  rewrite (rebuilt_code_body id (Code params mc ret exn_k md cbody) callee vs
             callee_vars Hun Hbind Hcvars).
  fold en'.
  split; [ exact Hbody | split; [ reflexivity | ] ].
  split.
  { unfold en'.
    cbn [env_kinds bind_variables bind_variables_then bind_continuations
         FlambdaReaperRebuild.empty_env rev].
    rewrite app_nil_r.
    rewrite (with_kinds_combine params); [ rewrite <- Hkp; exact Hok' | ].
    rewrite <- Hkp, length_map.
    exact (bind_params_length _ _ _ _ Hbind). }
  split.
  { (* The continuations of the code are the ways out of the call. *)
    assert (Hfree_c : ∀ k : continuation,
              k ∈ free_continuations_of_expr cbody -> k = ret ∨ k = exn_k).
    { intros k Hk. cbn [code_is_closed] in Hclosed_c.
      apply andb_true_iff in Hclosed_c as [Hcl _].
      apply andb_true_iff in Hcl as [_ Hcl].
      cbn [free_continuations_of_code] in Hcl.
      destruct (among k [ret; exn_k]) eqn:Hkin.
      - apply among_In in Hkin. destruct Hkin as [<- | [<- | []]]; auto.
      - exfalso.
        assert (Hin : k ∈ without [ret; exn_k]
                            (free_continuations_of_expr cbody)).
        { unfold without. apply filter_In. split; [ exact Hk | ].
          rewrite Hkin. reflexivity. }
        destruct (without [ret; exn_k] (free_continuations_of_expr cbody));
          [ destruct Hin | discriminate Hcl ]. }
    intros k Hk.
    pose proof (proj1 free_rebuild cbody Hcf_c en' k Hk) as Hk0.
    unfold code_conts, en'.
    cbn [code_exn_continuation code_return_continuation env_conts
         bind_variables bind_variables_then bind_continuations
         FlambdaReaperRebuild.empty_env app].
    destruct (Hfree_c k Hk0) as [-> | ->].
    - right. right. exists (unarize (cm_result_arity md)).
      split; [ cbn [lookup]; rewrite Pos.eqb_refl; reflexivity | ].
      exists (repeat true (length (unarize (cm_result_arity md)))),
        (cd_return cd).
      unfold resolve. cbn [lookup].
      rewrite Pos.eqb_refl.
      destruct (Pos.eqb_spec ret exn_k) as [E | _]; [ contradiction | ].
      split; [ right; split; [ reflexivity | apply forallb_repeat_true ] | ].
      split; [ apply repeat_length | ].
      split; [ reflexivity | exact Hret ].
    - left. cbn [lookup].
      destruct (Pos.eqb_spec exn_k ret) as [E | _];
        [ symmetry in E; contradiction | ].
      split; [ reflexivity | ].
      exists [cd_exn cd]. unfold resolve. cbn [lookup].
      rewrite Pos.eqb_refl.
      destruct (Pos.eqb_spec exn_k ret) as [E | _];
        [ symmetry in E; contradiction | ].
      split; [ reflexivity | exact Hway_exn' ]. }
  split; [ exact Hst | split; [ exact Hheap | ] ].
  split; [ | left; reflexivity ].
  split; [ exact Hlive' | ].
  split; [ | split; [ exact Hscr_c | split; [ exact Hcf_c | ] ] ].
  - exists []. split; [ split; [ intros k Hk; discriminate Hk
                              | intros r K _; reflexivity ] | ].
    apply (proj1 exits_bound_of_counted cbody _ [] _ Hcounted).
    intros k m Hl _. unfold code_conts.
    cbn [code_exn_continuation code_return_continuation lookup] in Hl |- *.
    destruct (Pos.eqb k exn_k);
      [ injection Hl as <-; exists [cd_exn cd]; split; reflexivity | ].
    destruct (Pos.eqb k ret); [ | discriminate Hl ].
    injection Hl as <-. exists (cd_return cd).
    split; [ reflexivity | exact Lr ].
  - split.
    2: { intros r K Hr. cbn [lookup] in Hr.
         destruct (Pos.eqb_spec r ret) as [-> | _]; [ | discriminate Hr ].
         injection Hr as <-. split; [ exact Hraise | ].
         unfold en'.
         cbn [env_kinds bind_variables bind_variables_then bind_continuations
              FlambdaReaperRebuild.empty_env rev].
         rewrite app_nil_r. apply Hrc. rewrite <- Hkp, length_map.
         pose proof (bind_params_length _ _ _ _ Hbind). lia. }
    split; [ exact Hfresh_c | ].
    split.
    + exists [].
      split; [ exact Hkinded_c | ].
      intros k m Hl _. unfold en' in Hl. cbn in Hl. discriminate Hl.
    + exists []. split; [ exact Hraises_c | ].
      intros k. unfold en'. cbn. exact I.
Qed.

(* A call that runs the code of its callee, whatever its kind: the first
   run enters the body of the code, and the second the rebuilt body,
   given the parameters and [my_closure] related, the code used, and the
   nodes of the continuations of the code transported to those of the
   call ([transports]). The continuations of the code are those of the
   call, or the wrapper that the rebuild gives it. *)
Lemma call_entered (ap : apply_expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
    (rcd rcd' : read_record)
    (en : rebuild_env) (ce : finite_map (list node)) (returns : list node)
    (exn : node) (ks : finite_map (list flambda_kind))
    (xs : finite_map bool) (id : code_id) (params : list variable)
    (mc : variable) (ret exn_k : continuation) (md : code_metadata)
    (cbody : expr) (cd : code_dep) (callee callee' : value)
    (slots' : list (value_slot * value)) (vs vs' extra extra' : list value)
    (callee_vars callee_vars' : env)
    (rets : finite_map (list flambda_kind)) :
  unchanged id (Code params mc ret exn_k md cbody) ->
  (∀ params : list node,
     lookup (ap_continuation ap) ce = Some params -> returns = params) ->
  (∀ (p : node) (params : list node),
     lookup (ap_exn_continuation ap) ce = Some (p :: params) ->
     exn = p
     ∧ simples_noted noted (exn_extra_simples ap)
     ∧ ∀ (q : node) (s : simple),
         (q, s) ∈ combine params (exn_extra_simples ap) ->
         g_alias G q (node_of_simple s)) ->
  env_ok (env_kinds en) vars vars' ->
  conts_related cont_rel over_rel rets
    (free_continuations_of_expr
       (make_apply_wrapper first_fresh en (rebuilt_call en ap)))
    conts conts' (env_conts en) ce ->
  (∀ (r : continuation) (K : list flambda_kind),
     lookup r rets = Some K ->
     ap_exn_continuation ap <> r
     ∧ (ap_continuation ap = r ->
        count_is ce r (length (unarize (ap_return_arity ap))) = true
        ∧ unarize (ap_return_arity ap) = K)) ->
  statics_rel statics statics' ->
  among (ap_exn_continuation ap) (map fst (env_conts en))
  || count_is ce (ap_exn_continuation ap)
       (S (length (ap_exn_extra_args ap))) = true ->
  among (ap_continuation ap) (map fst (env_conts en))
  || count_is ce (ap_continuation ap)
       (length (unarize (ap_return_arity ap))) = true ->
  match lookup (ap_continuation ap) ks with
  | Some expected => kinds_eqb expected (unarize (ap_return_arity ap))
  | None => true
  end = true ->
  kinds_agree rets (env_conts en) ks conts ->
  match lookup (ap_exn_continuation ap) xs with
  | Some b => b
  | None => no_exn_extra_args ap
  end = true ->
  exn_agree (env_conts en) xs ->
  ap_exn_continuation ap <> first_fresh ->
  eval_simples vars (exn_extra_simples ap) = Some extra ->
  eval_simples vars' (map (rewrite_simple sol en) (exn_extra_simples ap))
  = Some extra' ->
  map kind_of_value extra' = map kind_of_value extra ->
  flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc
    (State cbody callee_vars
       [(ret, resolve (ap_continuation ap) conts);
        (exn_k,
         with_extra_args (resolve (ap_exn_continuation ap) conts) extra)]
       statics hp rcd) ->
  eval_simple vars' (ap_callee ap) = Some callee' ->
  closure_of statics' callee' = Some (id, slots') ->
  rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap = Indirect
  ∨ rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap
    = Direct id ->
  lookup id (static_code statics')
  = Some (rebuild_c id (Code params mc ret exn_k md cbody)) ->
  cm_is_tupled md = false ->
  length (ap_args_arity ap) = length (cm_params_arity md) ->
  scrutinees_used G (Some id) cbody ->
  checked exact codes cbody = true ->
  raises_kept [] cbody = true ->
  code_is_closed (Code params mc ret exn_k md cbody) = true ->
  calls_kinded [] cbody = true ->
  among first_fresh (continuations_of cbody) = false ->
  ret <> exn_k -> binds_statics cbody = false ->
  conts_counted [(exn_k, 1); (ret, length (unarize (cm_result_arity md)))]
    cbody = true ->
  edges (Some id) [(exn_k, [cd_exn cd]); (ret, cd_return cd)] cbody ->
  length (cd_return cd) = length (unarize (cm_result_arity md)) ->
  eval_simples vars' (rebuild_arguments F sol en ap) = Some vs' ->
  map kind_of_value vs' = map kind_of_value vs ->
  map kind_of_value vs = unarize (ap_args_arity ap) ->
  unarize (ap_args_arity ap) = unarize (cm_params_arity md) ->
  kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
    (unarize_per_parameter (cm_params_arity md)) = true ->
  ap_return_arity ap = cm_result_arity md ->
  arity_eqb (ap_return_arity ap) (cm_result_arity md) = true ->
  bind_params params vs [(mc, callee)] = Some callee_vars ->
  bind_params params vs' [(mc, callee')] = Some callee_vars' ->
  env_ok (rev (combine params (map kind_of_value vs)) ++ [(mc, Value)])
    callee_vars callee_vars' ->
  au (Node_code_id id) ->
  transports (cd_exn cd) exn ->
  (∀ (i : nat) (n n' : node),
     nth_error returns i = Some n -> nth_error (cd_return cd) i = Some n' ->
     transports n' n) ->
  no_raise_to ret cbody = true ->
  (length params = length (unarize (cm_params_arity md)) ->
   rcalls_ok (rev (with_kinds params (unarize (cm_params_arity md)))
              ++ [(mc, Value)])
     ret (unarize (cm_result_arity md)) cbody = true) ->
  answered_step (State (Apply ap) vars conts statics hp rcd)
    (State cbody callee_vars
       [(ret, resolve (ap_continuation ap) conts);
        (exn_k,
         with_extra_args (resolve (ap_exn_continuation ap) conts) extra)]
       statics hp rcd)
    (State (make_apply_wrapper first_fresh en (rebuilt_call en ap)) vars'
       conts' statics' hp rcd').
Proof.
  intros Hun Hret_ce Hexn_ce Henv Hconts Hnr_ap Hst Hk_exn Hk_ret Hks Hagree
    Hxs
    Hxagree Hff_exn Hextra Hextra' Hkx Hflow' Hcallee' Hclosure' Hkind'
    Hcode' Htupled Hfull Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
    Hne Hbinds Hcounted Hbody Lr Hvs'' Hkinds' Hk1 Hk2 Hk2b Hk3 Hk3b Hbind
    Hbind' Hok' Hlive' Htr_exn Htr_ret Hraise_c Hrc_c.
  unfold answered_step, rebuilt_call in *.
  match goal with
  | |- context [make_apply_wrapper first_fresh en ?a] =>
    pose proof (make_apply_wrapper_free en a Hff_exn) as [Hret_free Hexn_free]
  end.
  cbn [ap_continuation ap_exn_continuation] in Hret_free, Hexn_free.
  assert (Hheap : hp = hp) by reflexivity.
  (* The exception continuation of the code: a way out, or a handler in
     scope that keeps the exception, with the extra arguments. *)
  assert (Hexn_to :
    ∃ extra'' : list value,
      eval_simples vars' (map fst (rebuild_exn_extra_args sol en ap))
      = Some extra''
      ∧ way_out cont_rel over_rel [cd_exn cd]
          (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
          (with_extra_args (resolve (ap_exn_continuation ap) conts')
             extra'')).
  { pose proof (Hxagree (ap_exn_continuation ap)) as Hxa.
    unfold rebuild_exn_extra_args.
    destruct (Hconts _ Hexn_free)
      as [(Hm_exn & nse & Hnse & Hway_exn)
         | [(mask1 & c1 & c1' & Hmask1 & Hc1 & Hc1' & Hce1 & Hcrel1 & _)
           | (Kx & HKx & _)]].
    3: { exfalso. exact (proj1 (Hnr_ap _ _ HKx) eq_refl). }
    - (* A way out: no extra argument. *)
      rewrite Hm_exn in Hxa |- *.
      destruct (lookup (ap_exn_continuation ap) xs) as [bx | ] eqn:Hxl;
        [ simpl in Hxa; destruct Hxa | ].
      assert (Hnox : ap_exn_extra_args ap = []).
      { unfold no_exn_extra_args in Hxs.
        destruct (ap_exn_extra_args ap); [ reflexivity | discriminate Hxs ]. }
      unfold exn_extra_simples in Hextra, Hextra'.
      rewrite Hnox in Hextra, Hextra' |- *.
      cbn [map eval_simples] in Hextra, Hextra' |- *.
      injection Hextra as <-. injection Hextra' as <-.
      exists []. split; [ reflexivity | ].
      cbn [with_extra_args].
      rewrite (lookup_none_among _ _ Hm_exn), Hnox in Hk_exn.
      cbn [orb length] in Hk_exn.
      unfold count_is in Hk_exn. rewrite Hnse in Hk_exn.
      apply Nat.eqb_eq in Hk_exn.
      destruct nse as [ | n1 [ | n2 nse]]; try discriminate Hk_exn.
      destruct (Hexn_ce n1 [] Hnse) as (-> & _).
      apply (way_out_transport _ _ [n1]); [ exact Hway_exn | reflexivity | ].
      intros i n n' Hn Hn'. destruct i as [ | [ | i]]; try discriminate Hn.
      injection Hn as <-. injection Hn' as <-.
      exact Htr_exn.
    - (* A handler in scope: an exception handler, which keeps the
         exception. *)
      rewrite Hmask1 in Hxa |- *.
      destruct (lookup (ap_exn_continuation ap) xs) as [bx | ] eqn:Hxl;
        [ | simpl in Hxa; destruct Hxa ].
      subst bx. pose proof (Hxa eq_refl) as Hhd.
      exists (filter_kept (tl mask1) extra'). split.
      + rewrite <- filter_kept_map, map_fst_rewrite.
        apply eval_simples_filter_kept. exact Hextra'.
      + right. right. left.
        exists (cd_exn cd), mask1, c1, c1', extra, extra'.
        split; [ reflexivity | ].
        split; [ unfold resolve; rewrite Hc1; reflexivity | ].
        split; [ unfold resolve; rewrite Hc1'; reflexivity | ].
        split; [ | split; [ exact Hcrel1 | split; [ exact Hhd | ] ] ].
        * intros i n p Hn Hp.
          destruct i as [ | [ | i]]; try discriminate Hn.
          injection Hn as <-.
          destruct (cont_params c1) as [ | p0 ps] eqn:Hcp;
            [ discriminate Hp | ].
          injection Hp as <-.
          destruct (Hexn_ce p0 ps Hce1) as (-> & _ & _).
          exact Htr_exn.
        * split; [ exact Hkx | ].
          intros i p v v' Hp Hv Hv' Hu.
          destruct (cont_params c1) as [ | p0 ps] eqn:Hcp;
            [ destruct i; discriminate Hp | ].
          cbn [nth_error] in Hp.
          destruct (Hexn_ce p0 ps Hce1) as (_ & Hnoted_x & Haliases_x).
          destruct (eval_simples_nth _ _ _ _ _ Hextra Hv) as (s & Hs & Hsv).
          assert (Hal : s_alias sol (Node_variable p) (node_of_simple s)
                        = true).
          { apply (alias_input _ _ _ Hc). apply Haliases_x.
            exact (nth_error_combine _ _ i _ _ Hp Hs). }
          destruct (rewrite_simples_eval en vars vars' Henv _ extra Hnoted_x
                      Hextra)
            as (extra2 & Hextra2 & _ & Hrels_x).
          rewrite Hextra' in Hextra2. injection Hextra2 as <-.
          apply (rel_alias G sol keep called (exact = true) True is_local Hc v
              _ _
                   v' Hal).
          apply (Hrels_x i s v v' Hs Hv Hv').
          destruct (is_var_used_cases p _ Hu) as [Hhu | Hk].
          -- apply kept_of_hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
          -- exact (kept_of_region en vars vars' s v Henv Hsv Hk). }
  destruct Hexn_to as (extra'' & Hextra'' & Hway_exn').
  match goal with
  | |- context [make_apply_wrapper first_fresh en ?a] => set (ap' := a) in *
  end.
  (* The calls of the second run, whatever they return to. *)
  assert (Hstep' : ∀ (k1 : continuation) (conts1 : cont_env),
            resolve (ap_exn_continuation ap) conts1
            = resolve (ap_exn_continuation ap) conts' ->
            step (State (Apply (with_continuation ap' k1)) vars' conts1
                    statics' hp rcd')
            = Step (State
                      (code_body
                         (rebuild_c id (Code params mc ret exn_k md cbody)))
                      callee_vars'
                      [(ret, resolve k1 conts1);
                       (exn_k,
                        with_extra_args
                          (resolve (ap_exn_continuation ap) conts') extra'')]
                      statics' hp rcd')).
  { intros k1 conts1 Hexn1. rewrite step_apply. unfold ap'.
    cbn [with_continuation ap_callee ap_args ap_call_kind ap_continuation
         ap_exn_continuation ap_exn_extra_args ap_args_arity ap_return_arity].
    rewrite Hcallee', Hvs''.
    unfold exn_extra_simples.
    cbn [with_continuation ap_exn_extra_args]. rewrite Hextra''.
    rewrite Hexn1.
    apply (call_rebuilt_code callee' _ id slots'
             (Code params mc ret exn_k md cbody) vs' (ap_args_arity ap)
             (ap_return_arity ap) _ _ statics' hp rcd' callee_vars' Hun
               Hclosure'
             Hkind' Hcode' Htupled Hfull).
    - apply rebuilt_code_no_statics; assumption.
    - apply kinds_eqb_eq. rewrite Hkinds'. exact Hk1.
    - exact Hk2b.
    - exact Hk3b.
    - exact Hbind'. }
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
    as (Hcvars & _).
  simpl_state.
  set (en' := bind_variables FlambdaReaperRebuild.empty_env
                ((mc, Value)
                 :: with_kinds params (unarize (cm_params_arity md)))).
  (* The callee, in the two runs, returning to a way out. *)
  assert (Hrel_callee : ∀ ret_to : cont_value,
            way_out cont_rel over_rel (cd_return cd)
              (resolve (ap_continuation ap) conts) ret_to ->
            related
              (State cbody callee_vars
                 [(ret, resolve (ap_continuation ap) conts);
                  (exn_k,
                   with_extra_args (resolve (ap_exn_continuation ap) conts)
                     extra)]
                 statics hp rcd)
              (State
                 (code_body (rebuild_c id (Code params mc ret exn_k md cbody)))
                 callee_vars'
                 [(ret, ret_to);
                  (exn_k,
                   with_extra_args (resolve (ap_exn_continuation ap) conts')
                     extra'')]
                 statics' hp rcd')).
  { intros ret_to Hway_ret.
    exact (code_entered id params mc ret exn_k md cbody cd callee vs
             callee_vars callee_vars' _ ret_to _ _ statics statics' hp rcd rcd'
               Hun
             Hflow'
             Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c Hne
             Hcounted Hbody Lr (eq_trans Hk1 Hk2) Hbind Hok' Hlive' Hway_ret
             Hway_exn'). }
  (* The callee, in the two runs, returning to the return continuation of
     code that changes. *)
  assert (Hrel_callee_ret : ∀ ret_to : cont_value,
            ret_rel cont_rel over_rel
              (repeat true (length (unarize (cm_result_arity md))))
              (unarize (cm_result_arity md)) (cd_return cd)
              (resolve (ap_continuation ap) conts) ret_to ->
            related
              (State cbody callee_vars
                 [(ret, resolve (ap_continuation ap) conts);
                  (exn_k,
                   with_extra_args (resolve (ap_exn_continuation ap) conts)
                     extra)]
                 statics hp rcd)
              (State
                 (code_body (rebuild_c id (Code params mc ret exn_k md cbody)))
                 callee_vars'
                 [(ret, ret_to);
                  (exn_k,
                   with_extra_args (resolve (ap_exn_continuation ap) conts')
                     extra'')]
                 statics' hp rcd')).
  { intros ret_to Hrr_ret.
    exact (code_entered_ret id params mc ret exn_k md cbody cd callee vs
             callee_vars callee_vars' _ ret_to _ _ statics statics' hp rcd rcd'
               Hun
             Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
             Hraise_c Hrc_c Hne Hcounted Hbody Lr (eq_trans Hk1 Hk2)
             Hbind Hok' Hlive' Hrr_ret Hway_exn'). }
  (* The return continuation of the call. *)
  destruct (Hconts _ Hret_free)
    as [(Hm_ret & ns & Hns & Hway_ret)
       | [(rmask & c0 & c0' & Hm_ret & Hc0 & Hc0' & Hce0 & Hcrel0 & Hrn0)
         | (Kx & HKx & mx & nsx & Hmx & Lmx & Hnsx & Hrrx)]].
  3: {
    (* The return continuation of code that changes: the callee returns
       there, directly or through the wrapper, with the mask of the
       return continuation. *)
    destruct (proj2 (Hnr_ap _ _ HKx) eq_refl) as [Hcnt HK].
    unfold count_is in Hcnt. rewrite Hnsx in Hcnt.
    apply Nat.eqb_eq in Hcnt.
    pose proof (Hret_ce _ Hnsx) as Hrn. subst returns.
    assert (HKc : unarize (cm_result_arity md) = Kx)
      by (rewrite <- Hk3; exact HK).
    assert (Lcd : length (cd_return cd) = length nsx)
      by (rewrite Lr, Hcnt, Hk3; reflexivity).
    unfold make_apply_wrapper.
    change (ap_continuation ap') with (ap_continuation ap).
    destruct Hmx as [Hmx | [Hmx Hall]].
    2: { rewrite Hmx. right. left. eexists.
         split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
         apply Hrel_callee_ret. rewrite HKc.
         apply (ret_rel_all_true _ _ mx Kx nsx);
           [ exact Hrrx | exact Hall | exact Lmx | exact Lcd | ].
         intros i n n' Hn Hn'. exact (Htr_ret i n n' Hn Hn'). }
    rewrite Hmx.
    destruct (forallb (fun keep => keep) mx) eqn:Hall.
    - right. left. eexists.
      split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
      apply Hrel_callee_ret. rewrite HKc.
      apply (ret_rel_all_true _ _ mx Kx nsx);
        [ exact Hrrx | exact Hall | exact Lmx | exact Lcd | ].
      intros i n n' Hn Hn'. exact (Htr_ret i n n' Hn Hn').
    - right. right. right. eexists. exists 0. cbv zeta. split.
      + cbn [run]. rewrite step_let_cont_non_recursive. cbv beta iota.
        rewrite (Hstep' first_fresh).
        * reflexivity.
        * unfold resolve. cbn [lookup].
          apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity.
      + apply Hrel_callee_ret. rewrite HKc.
        match goal with
        | |- ret_rel _ _ _ _ _ _ ?w =>
          replace w with (wrapper (ap_continuation ap) mx Kx vars' conts')
        end.
        * apply (ret_rel_plain_wrapper _ _ mx Kx (cd_return cd) nsx);
            [ exact Hrrx | exact Lmx | exact (eq_sym Lcd) | | exact Hall
            | ].
          -- rewrite Lr, HKc. reflexivity.
          -- intros i n n' Hn Hn'. exact (Htr_ret i n' n Hn' Hn).
        * unfold resolve, wrapper. cbn [lookup]. rewrite Pos.eqb_refl, <- HK.
          reflexivity. }
  - (* Not a handler of the expression: the code returns there. *)
    pose proof Hk_ret as Hcount_ret.
    rewrite (lookup_none_among _ _ Hm_ret) in Hcount_ret.
    cbn [orb] in Hcount_ret.
    unfold count_is in Hcount_ret. rewrite Hns in Hcount_ret.
    apply Nat.eqb_eq in Hcount_ret.
    pose proof (Hret_ce ns Hns) as Hrn. subst returns.
    assert (Hway' : way_out cont_rel over_rel (cd_return cd)
                      (resolve (ap_continuation ap) conts)
                      (resolve (ap_continuation ap) conts')).
    { apply (way_out_transport _ _ ns); [ exact Hway_ret | | ].
      - rewrite Lr, Hcount_ret, Hk3. reflexivity.
      - intros i n n' Hn Hn'.
        exact (Htr_ret i n n' Hn Hn'). }
    right. left. eexists. split.
    + unfold make_apply_wrapper.
      change (ap_continuation ap') with (ap_continuation ap).
      rewrite Hm_ret.
      exact (Hstep' (ap_continuation ap) conts' eq_refl).
    + apply Hrel_callee. exact Hway'.
  - (* A handler of the expression: directly or through the wrapper. *)
    pose proof (Hret_ce _ Hce0) as Hrn. subst returns.
    assert (Hres0 : resolve (ap_continuation ap) conts = c0)
      by (unfold resolve; rewrite Hc0; reflexivity).
    assert (Hret0 : returns_to (cd_return cd) c0).
    { intros i n p Hn Hp.
      exact (Htr_ret i p n Hp Hn). }
    assert (Hkc0 : cont_kinds c0 = unarize (ap_return_arity ap)).
    { destruct (Hagree _ _ Hm_ret
                  Hrn0)
        as (c2 & Hc2 & Hks2).
      rewrite Hc0 in Hc2. injection Hc2 as <-.
      rewrite Hks2 in Hks. apply kinds_eqb_eq in Hks. exact Hks. }
    unfold make_apply_wrapper.
    change (ap_continuation ap') with (ap_continuation ap).
    rewrite Hm_ret.
    destruct (forallb (fun keep => keep) rmask) eqn:Hall.
    + right. left. eexists.
      split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
      apply Hrel_callee. right. left. exists rmask, c0'. rewrite Hres0.
      split; [ exact Hret0 | split; [ exact Hcrel0 | left ] ].
      split; [ unfold resolve; rewrite Hc0'; reflexivity | exact Hall ].
    + right. right. right. eexists. exists 0. cbv zeta. split.
      * cbn [run]. rewrite step_let_cont_non_recursive. cbv beta iota.
        rewrite (Hstep' first_fresh).
        -- reflexivity.
        -- unfold resolve. cbn [lookup].
           apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity.
      * apply Hrel_callee. right. left. exists rmask, c0'. rewrite Hres0.
        split; [ exact Hret0 | split; [ exact Hcrel0 | right ] ].
        exists (ap_continuation ap), vars', conts'.
        split; [ | exact Hc0' ].
        unfold resolve, wrapper. cbn [lookup]. rewrite Pos.eqb_refl, Hkc0.
        reflexivity.
Qed.

(* The call of [rebuild_apply_cc] that changes its calling convention. *)
Definition cc_call (en : rebuild_env) (ap : apply_expr)
    (pd rd : list param_decision) : apply_expr :=
  let groups :=
    map (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
      (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
         (group_by_parameter (ap_args_arity ap) pd))
  in
  {| ap_callee := ap_callee ap;
     ap_call_kind :=
       rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap;
     ap_args := flat_map (map fst) groups;
     ap_continuation := ap_continuation ap;
     ap_exn_continuation := ap_exn_continuation ap;
     ap_exn_extra_args := rebuild_exn_extra_args sol en ap;
     ap_args_arity := arity_of_groups groups;
     ap_return_arity := arity_of_decisions rd |}.

Lemma make_apply_wrapper_cc_free (en : rebuild_env) (ap : apply_expr)
    (fd : list (param_decision * flambda_kind)) :
  ap_exn_continuation ap <> first_fresh ->
  ap_continuation ap
    ∈ free_continuations_of_expr (make_apply_wrapper_cc first_fresh en ap fd)
  ∧ ap_exn_continuation ap
    ∈ free_continuations_of_expr (make_apply_wrapper_cc first_fresh en ap fd).
Proof.
  intros Hne. unfold make_apply_wrapper_cc. cbv zeta.
  destruct (forallb _ _);
    cbn [free_continuations_of_expr free_continuations_of_handler
         apply_cont_continuations ac_continuation ac_trap_action
         with_continuation ap_continuation ap_exn_continuation];
    try (split; [ left; reflexivity | right; left; reflexivity ]).
  split; [ left; reflexivity | ].
  right. unfold without. apply filter_In.
  split; [ right; left; reflexivity | ].
  rewrite among_cons. apply Pos.eqb_neq in Hne. rewrite Hne. reflexivity.
Qed.

(* A wrapper is not made where the mask of the continuation is that of
   what the callee keeps. *)
Lemma mask_of_forallb (rd : list param_decision) :
  ∀ (amask : list bool) (kinds : list flambda_kind),
    length amask = length rd -> length rd = length kinds ->
    forallb (fun '(keep, (decision, _)) => Bool.eqb keep (is_kept decision))
      (combine amask (combine rd kinds))
    = true ->
    map is_kept rd = amask.
Proof.
  induction rd as [ | d rd IH]; intros [ | b amask] [ | kind kinds] La Lk H;
    try discriminate La; try discriminate Lk; [ reflexivity | ].
  cbn [combine forallb] in H. apply andb_true_iff in H as [H1 H2].
  apply Bool.eqb_prop in H1. cbn [map]. rewrite H1.
  f_equal. exact (IH amask kinds ltac:(simpl in La; lia)
                    ltac:(simpl in Lk; lia) H2).
Qed.

Lemma pad_mask_kept (rd : list param_decision) :
  pad_mask (length rd) (map is_kept rd) = map is_kept rd.
Proof. rewrite <- (length_map is_kept rd). apply pad_mask_self. Qed.

Lemma mask_forallb_self (rd : list param_decision) :
  ∀ kinds : list flambda_kind,
    forallb (fun '(keep, (decision, _)) => Bool.eqb keep (is_kept decision))
      (combine (map is_kept rd) (combine rd kinds))
    = true.
Proof.
  induction rd as [ | d rd IH]; intros [ | kind kinds]; [ reflexivity .. | ].
  cbn [map combine forallb]. rewrite Bool.eqb_reflx. exact (IH kinds).
Qed.

Lemma some_delete (rd : list param_decision) :
  ∀ kinds : list flambda_kind,
    length rd = length kinds ->
    forallb (fun '(keep, (decision, _)) => Bool.eqb keep (is_kept decision))
      (combine (repeat true (length rd)) (combine rd kinds))
    = false ->
    ∃ i : nat, nth_error rd i = Some Delete.
Proof.
  induction rd as [ | d rd IH]; intros [ | kind kinds] L H;
    try discriminate L; [ discriminate H | ].
  cbn [length repeat combine forallb] in H.
  apply andb_false_iff in H as [H | H].
  - exists 0. destruct d; [ discriminate H | reflexivity ].
  - destruct (IH kinds ltac:(simpl in L; lia) H) as (i & Hi).
    exists (S i). exact Hi.
Qed.

Lemma mask_mismatch (rd : list param_decision) :
  ∀ (m0 : list bool) (kinds : list flambda_kind),
    length m0 = length rd -> length rd = length kinds ->
    forallb (fun '(keep, (decision, _)) => Bool.eqb keep (is_kept decision))
      (combine m0 (combine rd kinds))
    = false ->
    ∃ i : nat, nth_error m0 i = Some false ∨ nth_error rd i = Some Delete.
Proof.
  induction rd as [ | d rd IH]; intros [ | b m0] [ | kind kinds] Lm L H;
    try discriminate Lm; try discriminate L; [ discriminate H | ].
  cbn [combine forallb] in H.
  apply andb_false_iff in H as [H | H].
  - exists 0. cbn [nth_error].
    destruct d; [ | right; reflexivity ].
    left. destruct b; [ discriminate H | reflexivity ].
  - destruct (IH m0 kinds ltac:(simpl in Lm; lia) ltac:(simpl in L; lia) H)
      as (i & Hi).
    exists (S i). exact Hi.
Qed.

Lemma unarize_singletons (ks : list flambda_kind) :
  unarize_component (Unboxed_product (map Singleton ks)) = ks.
Proof.
  induction ks as [ | k ks IH]; [ reflexivity | ].
  cbn [map unarize_component flat_map app] in *. rewrite IH. reflexivity.
Qed.

Lemma unarize_grouped (gs : list (list param_decision)) :
  unarize (arity_of_grouped_decisions gs) = kept_kinds (concat gs).
Proof.
  induction gs as [ | g gs IH]; [ reflexivity | ].
  unfold unarize, arity_of_grouped_decisions in *. cbn [map flat_map concat].
  rewrite IH, unarize_singletons, kept_kinds_app. reflexivity.
Qed.

(* A direct call of code that changes its calling convention: the first
   run enters the body of the code, and the second the rebuilt body,
   with the parameters kept related, and [my_closure]; the call returns
   to a handler, directly or through the wrapper of
   [make_apply_wrapper_cc]. *)
Lemma call_entered_cc (ap : apply_expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
    (rcd rcd' : read_record)
    (en : rebuild_env) (ce : finite_map (list node)) (returns : list node)
    (exn : node) (ks : finite_map (list flambda_kind))
    (xs : finite_map bool) (id : code_id) (params : list variable)
    (mc : variable) (ret exn_k : continuation) (md : code_metadata)
    (cbody : expr) (cd : code_dep) (pd rd : list param_decision)
    (callee callee' : value)
    (slots' : list (value_slot * value)) (vs vs' extra extra' : list value)
    (callee_vars callee_vars' : env)
    (rets : finite_map (list flambda_kind)) :
  changing id (Code params mc ret exn_k md cbody) pd rd ->
  (∀ params : list node,
     lookup (ap_continuation ap) ce = Some params -> returns = params) ->
  (∀ (p : node) (params : list node),
     lookup (ap_exn_continuation ap) ce = Some (p :: params) ->
     exn = p
     ∧ simples_noted noted (exn_extra_simples ap)
     ∧ ∀ (q : node) (s : simple),
         (q, s) ∈ combine params (exn_extra_simples ap) ->
         g_alias G q (node_of_simple s)) ->
  env_ok (env_kinds en) vars vars' ->
  conts_related cont_rel over_rel rets
    (free_continuations_of_expr
       (make_apply_wrapper_cc first_fresh en (cc_call en ap pd rd)
          (combine rd (unarize (ap_return_arity ap)))))
    conts conts' (env_conts en) ce ->
  (∀ (r : continuation) (K : list flambda_kind),
     lookup r rets = Some K ->
     ap_exn_continuation ap <> r
     ∧ (ap_continuation ap = r ->
        count_is ce r (length (unarize (ap_return_arity ap))) = true
        ∧ unarize (ap_return_arity ap) = K)) ->
  statics_rel statics statics' ->
  among (ap_exn_continuation ap) (map fst (env_conts en))
  || count_is ce (ap_exn_continuation ap)
       (S (length (ap_exn_extra_args ap))) = true ->
  among (ap_continuation ap) (map fst (env_conts en))
  || count_is ce (ap_continuation ap)
       (length (unarize (ap_return_arity ap))) = true ->
  match lookup (ap_continuation ap) ks with
  | Some expected => kinds_eqb expected (unarize (ap_return_arity ap))
  | None => true
  end = true ->
  kinds_agree rets (env_conts en) ks conts ->
  match lookup (ap_exn_continuation ap) xs with
  | Some b => b
  | None => no_exn_extra_args ap
  end = true ->
  exn_agree (env_conts en) xs ->
  ap_exn_continuation ap <> first_fresh ->
  eval_simples vars (exn_extra_simples ap) = Some extra ->
  eval_simples vars' (map (rewrite_simple sol en) (exn_extra_simples ap))
  = Some extra' ->
  map kind_of_value extra' = map kind_of_value extra ->
  flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc
    (State cbody callee_vars
       [(ret, resolve (ap_continuation ap) conts);
        (exn_k,
         with_extra_args (resolve (ap_exn_continuation ap) conts) extra)]
       statics hp rcd) ->
  eval_simple vars' (ap_callee ap) = Some callee' ->
  closure_of statics' callee' = Some (id, slots') ->
  rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap = Indirect
  ∨ rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap
    = Direct id ->
  lookup id (static_code statics')
  = Some (rebuild_c id (Code params mc ret exn_k md cbody)) ->
  scrutinees_used G (Some id) cbody ->
  checked exact codes cbody = true ->
  raises_kept [] cbody = true ->
  code_is_closed (Code params mc ret exn_k md cbody) = true ->
  calls_kinded [] cbody = true ->
  among first_fresh (continuations_of cbody) = false ->
  no_raise_to ret cbody = true ->
  (length params = length (unarize (cm_params_arity md)) ->
   rcalls_ok (rev (with_kinds params (unarize (cm_params_arity md)))
              ++ [(mc, Value)])
     ret (unarize (cm_result_arity md)) cbody = true) ->
  ret <> exn_k -> binds_statics cbody = false ->
  conts_counted [(exn_k, 1); (ret, length (unarize (cm_result_arity md)))]
    cbody = true ->
  edges (Some id) [(exn_k, [cd_exn cd]); (ret, cd_return cd)] cbody ->
  length (cd_return cd) = length (unarize (cm_result_arity md)) ->
  length (cd_params cd) = length (unarize (cm_params_arity md)) ->
  pd = decisions_of sol (cd_params cd) (unarize (cm_params_arity md)) ->
  rd = decisions_of sol (cd_return cd) (unarize (cm_result_arity md)) ->
  eval_simples vars' (map (rewrite_simple sol en) (ap_args ap)) = Some vs' ->
  map kind_of_value vs' = map kind_of_value vs ->
  map kind_of_value vs = unarize (ap_args_arity ap) ->
  unarize_per_parameter (ap_args_arity ap)
  = unarize_per_parameter (cm_params_arity md) ->
  ap_return_arity ap = cm_result_arity md ->
  length (ap_args ap) = length (unarize (ap_args_arity ap)) ->
  bind_params params vs [(mc, callee)] = Some callee_vars ->
  bind_params (filter_kept (map is_kept pd) params)
    (filter_kept (map is_kept pd) vs') [(mc, callee')]
  = Some callee_vars' ->
  env_ok (rev (combine params (map kind_of_value vs)) ++ [(mc, Value)])
    callee_vars callee_vars' ->
  au (Node_code_id id) ->
  transports (cd_exn cd) exn ->
  (∀ (i : nat) (n n' : node),
     nth_error returns i = Some n -> nth_error (cd_return cd) i = Some n' ->
     transports n' n) ->
  answered_step (State (Apply ap) vars conts statics hp rcd)
    (State cbody callee_vars
       [(ret, resolve (ap_continuation ap) conts);
        (exn_k,
         with_extra_args (resolve (ap_exn_continuation ap) conts) extra)]
       statics hp rcd)
    (State
       (make_apply_wrapper_cc first_fresh en (cc_call en ap pd rd)
          (combine rd (unarize (ap_return_arity ap))))
       vars' conts' statics' hp rcd').
Proof.
  intros Hch Hret_ce Hexn_ce Henv Hconts Hnr_ap Hst Hk_exn Hk_ret Hks Hagree
    Hxs Hxagree Hff_exn Hextra Hextra' Hkx Hflow' Hcallee' Hclosure' Hkind'
    Hcode' Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c Hraise_c Hrc_c
    Hne Hbinds Hcounted Hbody Lr Lp Hpd Hrd Hvs'' Hkinds' Hk1 Hk2 Hk3 La
    Hbind Hbind' Hok' Hlive' Htr_exn Htr_ret.
  unfold answered_step in *.
  set (fd := combine rd (unarize (ap_return_arity ap))) in *.
  set (ap' := cc_call en ap pd rd) in *.
  pose proof (make_apply_wrapper_cc_free en ap' fd Hff_exn)
    as [Hret_free Hexn_free].
  change (ap_continuation ap') with (ap_continuation ap) in Hret_free.
  change (ap_exn_continuation ap') with (ap_exn_continuation ap) in Hexn_free.
  assert (Hheap : hp = hp) by reflexivity.
  assert (Lpd : length pd = length (unarize (cm_params_arity md))).
  { rewrite Hpd, length_decisions_of, Lp. apply Nat.min_id. }
  assert (Lrd : length rd = length (unarize (cm_result_arity md))).
  { rewrite Hrd, length_decisions_of, Lr. apply Nat.min_id. }
  assert (Lfd : length fd = length rd).
  { unfold fd. rewrite length_combine, Hk3, Lrd. apply Nat.min_id. }
  (* The exception continuation of the code: a way out, or a handler in
     scope that keeps the exception, with the extra arguments. *)
  assert (Hexn_to :
    ∃ extra'' : list value,
      eval_simples vars' (map fst (rebuild_exn_extra_args sol en ap))
      = Some extra''
      ∧ way_out cont_rel over_rel [cd_exn cd]
          (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
          (with_extra_args (resolve (ap_exn_continuation ap) conts')
             extra'')).
  { pose proof (Hxagree (ap_exn_continuation ap)) as Hxa.
    unfold rebuild_exn_extra_args.
    destruct (Hconts _ Hexn_free)
      as [(Hm_exn & nse & Hnse & Hway_exn)
         | [(mask1 & c1 & c1' & Hmask1 & Hc1 & Hc1' & Hce1 & Hcrel1 & _)
           | (Kx & HKx & _)]].
    3: { exfalso. exact (proj1 (Hnr_ap _ _ HKx) eq_refl). }
    - (* A way out: no extra argument. *)
      rewrite Hm_exn in Hxa |- *.
      destruct (lookup (ap_exn_continuation ap) xs) as [bx | ] eqn:Hxl;
        [ simpl in Hxa; destruct Hxa | ].
      assert (Hnox : ap_exn_extra_args ap = []).
      { unfold no_exn_extra_args in Hxs.
        destruct (ap_exn_extra_args ap); [ reflexivity | discriminate Hxs ]. }
      unfold exn_extra_simples in Hextra, Hextra'.
      rewrite Hnox in Hextra, Hextra' |- *.
      cbn [map eval_simples] in Hextra, Hextra' |- *.
      injection Hextra as <-. injection Hextra' as <-.
      exists []. split; [ reflexivity | ].
      cbn [with_extra_args].
      rewrite (lookup_none_among _ _ Hm_exn), Hnox in Hk_exn.
      cbn [orb length] in Hk_exn.
      unfold count_is in Hk_exn. rewrite Hnse in Hk_exn.
      apply Nat.eqb_eq in Hk_exn.
      destruct nse as [ | n1 [ | n2 nse]]; try discriminate Hk_exn.
      destruct (Hexn_ce n1 [] Hnse) as (-> & _).
      apply (way_out_transport _ _ [n1]); [ exact Hway_exn | reflexivity | ].
      intros i n n' Hn Hn'. destruct i as [ | [ | i]]; try discriminate Hn.
      injection Hn as <-. injection Hn' as <-.
      exact Htr_exn.
    - (* A handler in scope: an exception handler, which keeps the
         exception. *)
      rewrite Hmask1 in Hxa |- *.
      destruct (lookup (ap_exn_continuation ap) xs) as [bx | ] eqn:Hxl;
        [ | simpl in Hxa; destruct Hxa ].
      subst bx. pose proof (Hxa eq_refl) as Hhd.
      exists (filter_kept (tl mask1) extra'). split.
      + rewrite <- filter_kept_map, map_fst_rewrite.
        apply eval_simples_filter_kept. exact Hextra'.
      + right. right. left.
        exists (cd_exn cd), mask1, c1, c1', extra, extra'.
        split; [ reflexivity | ].
        split; [ unfold resolve; rewrite Hc1; reflexivity | ].
        split; [ unfold resolve; rewrite Hc1'; reflexivity | ].
        split; [ | split; [ exact Hcrel1 | split; [ exact Hhd | ] ] ].
        * intros i n p Hn Hp.
          destruct i as [ | [ | i]]; try discriminate Hn.
          injection Hn as <-.
          destruct (cont_params c1) as [ | p0 ps] eqn:Hcp;
            [ discriminate Hp | ].
          injection Hp as <-.
          destruct (Hexn_ce p0 ps Hce1) as (-> & _ & _).
          exact Htr_exn.
        * split; [ exact Hkx | ].
          intros i p v v' Hp Hv Hv' Hu.
          destruct (cont_params c1) as [ | p0 ps] eqn:Hcp;
            [ destruct i; discriminate Hp | ].
          cbn [nth_error] in Hp.
          destruct (Hexn_ce p0 ps Hce1) as (_ & Hnoted_x & Haliases_x).
          destruct (eval_simples_nth _ _ _ _ _ Hextra Hv) as (s & Hs & Hsv).
          assert (Hal : s_alias sol (Node_variable p) (node_of_simple s)
                        = true).
          { apply (alias_input _ _ _ Hc). apply Haliases_x.
            exact (nth_error_combine _ _ i _ _ Hp Hs). }
          destruct (rewrite_simples_eval en vars vars' Henv _ extra Hnoted_x
                      Hextra)
            as (extra2 & Hextra2 & _ & Hrels_x).
          rewrite Hextra' in Hextra2. injection Hextra2 as <-.
          apply (rel_alias G sol keep called (exact = true) True is_local Hc v
              _ _
                   v' Hal).
          apply (Hrels_x i s v v' Hs Hv Hv').
          destruct (is_var_used_cases p _ Hu) as [Hhu | Hk].
          -- apply kept_of_hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
          -- exact (kept_of_region en vars vars' s v Henv Hsv Hk). }
  destruct Hexn_to as (extra'' & Hextra'' & Hway_exn').
  destruct (cc_call_args en ap md pd Hk2 La Lpd) as [Hargs_eq Harity_eq].
  destruct (code_fields_rebuild_cc id (Code params mc ret exn_k md cbody) pd
              rd Hch)
    as (Ep & Em & Er & Ee & Epa & Era & Et).
  cbn [code_params code_my_closure code_return_continuation
       code_exn_continuation code_params_arity] in Ep, Em, Er, Ee, Epa.
  (* The calls of the second run, whatever they return to. *)
  assert (Hstep' : ∀ (k1 : continuation) (conts1 : cont_env),
            resolve (ap_exn_continuation ap) conts1
            = resolve (ap_exn_continuation ap) conts' ->
            step (State (Apply (with_continuation ap' k1)) vars' conts1
                    statics' hp rcd')
            = Step (State
                      (code_body
                         (rebuild_c id (Code params mc ret exn_k md cbody)))
                      callee_vars'
                      [(ret, resolve k1 conts1);
                       (exn_k,
                        with_extra_args
                          (resolve (ap_exn_continuation ap) conts') extra'')]
                      statics' hp rcd')).
  { intros k1 conts1 Hexn1. rewrite step_apply. unfold ap', cc_call.
    cbn [with_continuation ap_callee ap_args ap_call_kind ap_continuation
         ap_exn_continuation ap_exn_extra_args ap_args_arity ap_return_arity].
    rewrite Hcallee', Hargs_eq, (eval_simples_filter_kept vars' _ _ _ Hvs'').
    unfold exn_extra_simples.
    cbn [with_continuation ap_exn_extra_args]. rewrite Hextra''.
    rewrite Hexn1, Harity_eq, <- Epa.
    apply (call_rebuilt_code_cc callee' _ id slots'
             (Code params mc ret exn_k md cbody) pd rd
             (filter_kept (map is_kept pd) vs') _ _ statics' hp rcd'
               callee_vars'
             Hch Hclosure' Hkind' Hcode').
    - apply rebuilt_code_no_statics; assumption.
    - apply kinds_eqb_eq. rewrite Epa, <- filter_kept_map, Hkinds', Hk1.
      rewrite (unarize_per_parameter_eq _ _ Hk2).
      rewrite unarize_grouped. unfold group_by_parameter.
      rewrite group_by_sizes_concat; [ | rewrite Lpd; symmetry;
                                         apply sum_arity_sizes ].
      rewrite Hpd. symmetry.
      apply kept_kinds_decisions_of. exact Lp.
    - exact Hbind'. }
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hflow')
    as (Hcvars & _).
  simpl_state.
  set (en' := bind_variables FlambdaReaperRebuild.empty_env
                ((mc, Value)
                 :: with_kinds params (unarize (cm_params_arity md)))).
  (* The callee, in the two runs, returning where the call does. *)
  assert (Hrel_callee : ∀ ret_to : cont_value,
            ret_rel cont_rel over_rel (map is_kept rd)
              (unarize (cm_result_arity md)) (cd_return cd)
              (resolve (ap_continuation ap) conts) ret_to ->
            related
              (State cbody callee_vars
                 [(ret, resolve (ap_continuation ap) conts);
                  (exn_k,
                   with_extra_args (resolve (ap_exn_continuation ap) conts)
                     extra)]
                 statics hp rcd)
              (State
                 (code_body (rebuild_c id (Code params mc ret exn_k md cbody)))
                 callee_vars'
                 [(ret, ret_to);
                  (exn_k,
                   with_extra_args (resolve (ap_exn_continuation ap) conts')
                     extra'')]
                 statics' hp rcd')).
  { intros ret_to Hrr.
    assert (Lpar : length params = length (unarize (cm_params_arity md))).
    { rewrite <- (bind_params_length _ _ _ _ Hbind),
        <- (unarize_per_parameter_eq _ _ Hk2), <- Hk1, length_map.
      reflexivity. }
    exact (code_entered_cc id params mc ret exn_k md cbody cd pd rd callee
             vs callee_vars callee_vars' _ ret_to _ _ statics statics' hp rcd
               rcd' Hch
             Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
             Hraise_c (Hrc_c Lpar) Hne Hcounted Hbody Lr
             (eq_trans Hk1 (unarize_per_parameter_eq _ _ Hk2)) Hbind Hok'
             Hlive' Lrd Hrr Hway_exn'). }
  assert (Hkeep : ∀ (i : nat) (kind : flambda_kind),
            nth_error rd i = Some (Keep kind) ->
            nth_error (unarize (ap_return_arity ap)) i = Some kind).
  { intros i kind Hi. rewrite Hrd in Hi.
    destruct (decisions_of_nth _ _ _ _ _ Hi) as (n & kind' & _ & Hk' & Hd).
    rewrite Hk3. destruct (is_node_used sol n kind'); [ | discriminate Hd ].
    injection Hd as <-. exact Hk'. }
  (* A result that the code no longer returns has no use. *)
  assert (Hdel0 : ∀ i : nat,
            nth_error rd i = Some Delete ->
            ∃ n : node,
              nth_error (cd_return cd) i = Some n ∧ ~ hu n
              ∧ ∀ kind : flambda_kind,
                  nth_error (unarize (ap_return_arity ap)) i = Some kind ->
                  kind <> Region ∧ kind <> Rec_info).
  { intros i Hi. rewrite Hrd in Hi.
    destruct (decisions_of_nth _ _ _ _ _ Hi) as (n & kind' & Hn & Hk' & Hd).
    exists n. split; [ exact Hn | ].
    destruct (is_node_used sol n kind') eqn:Hu; [ discriminate Hd | ].
    unfold is_node_used in Hu.
    split.
    - intros Hhu. destruct kind'; try discriminate Hu; congruence.
    - intros kind Hkind. rewrite Hk3, Hk' in Hkind. injection Hkind as <-.
      destruct kind'; try discriminate Hu; split; discriminate. }
  unfold make_apply_wrapper_cc. cbv zeta.
  change (ap_continuation ap') with (ap_continuation ap).
  unfold apply_decisions.
  destruct (Hconts _ Hret_free)
    as [(Hm_ret & ns & Hns & Hway_ret)
       | [(rmask & c0 & c0' & Hm_ret & Hc0 & Hc0' & Hce0 & Hcrel0 & Hrn0)
         | (Kr & HKr & m0 & ns0 & Hm0 & Hlm0 & Hns0 & Hrr0)]].
  - (* A way out: directly if the code keeps every result, otherwise
       through the wrapper, which gives the way out poison where the code
       no longer returns a result, which has no use there. *)
    pose proof Hk_ret as Hcount_ret.
    rewrite (lookup_none_among _ _ Hm_ret) in Hcount_ret.
    cbn [orb] in Hcount_ret.
    unfold count_is in Hcount_ret. rewrite Hns in Hcount_ret.
    apply Nat.eqb_eq in Hcount_ret.
    pose proof (Hret_ce ns Hns) as Hrn. subst returns.
    assert (Lns : length ns = length (cd_return cd))
      by (rewrite Lr, Hcount_ret, Hk3; reflexivity).
    rewrite Hm_ret, Lfd.
    change (firstn (length rd) ([] ++ repeat true (length rd)))
      with (pad_mask (length rd) []).
    destruct (forallb (fun '(keep, (decision, _)) =>
                         Bool.eqb keep (is_kept decision))
                (combine (pad_mask (length rd) []) fd)) eqn:Hall.
    + right. left. eexists.
      split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
      apply Hrel_callee. apply Ret_base. left. split.
      * unfold fd in Hall.
        assert (Lk : length rd = length (unarize (ap_return_arity ap)))
          by (rewrite Lrd, Hk3; reflexivity).
        rewrite (mask_of_forallb rd _ _ (length_pad_mask _ _) Lk Hall).
        rewrite pad_mask_nil. apply forallb_repeat_true.
      * apply (way_out_transport _ _ ns);
          [ exact Hway_ret | exact (eq_sym Lns) | ].
        intros i n n' Hn Hn'. exact (Htr_ret i n n' Hn Hn').
    + right. right. right. eexists. exists 0. cbv zeta. split.
      * cbn [run]. rewrite step_let_cont_non_recursive. cbv beta iota.
        rewrite (Hstep' first_fresh).
        -- reflexivity.
        -- unfold resolve. cbn [lookup].
           apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity.
      * apply Hrel_callee. apply Ret_base. right. right. right.
        exists (ap_continuation ap), rd, vars', conts', ns.
        split.
        { unfold resolve, cc_wrapper, fd. cbn [lookup].
          rewrite Pos.eqb_refl, Hk3. reflexivity. }
        split; [ exact Hway_ret | split; [ reflexivity | ] ].
        split; [ exact Lrd | split; [ exact Lns | ] ].
        split; [ rewrite Lr; symmetry; exact Lrd | ].
        split.
        { unfold fd in Hall. rewrite pad_mask_nil in Hall.
          apply (some_delete rd (unarize (ap_return_arity ap)));
            [ rewrite Lrd, Hk3; reflexivity | exact Hall ]. }
        split; [ intros i n n' Hn Hn'; exact (Htr_ret i n' n Hn' Hn) | ].
        split; [ rewrite <- Hk3; exact Hkeep | ].
        intros i Hi. destruct (Hdel0 i Hi) as (n & Hn & Hnu & Hkind).
        split; [ | rewrite <- Hk3; exact Hkind ].
        intros n' Hn' Hu'. apply Hnu.
        exact (proj1 (Htr_ret i n' n Hn' Hn) Hu').
  - (* A handler of the expression: directly or through the wrapper. *)
    pose proof (Hret_ce _ Hce0) as Hrn. subst returns.
    assert (Hres0 : resolve (ap_continuation ap) conts = c0)
      by (unfold resolve; rewrite Hc0; reflexivity).
    assert (Hret0 : returns_to (cd_return cd) c0).
    { intros i n p Hn Hp.
      exact (Htr_ret i p n Hp Hn). }
    assert (Hdel : ∀ (i : nat) (p : variable) (kind : flambda_kind),
              nth_error rd i = Some Delete ->
              nth_error (cont_params c0) i = Some (Node_variable p) ->
              nth_error (unarize (ap_return_arity ap)) i = Some kind ->
              is_var_used sol p kind = false).
    { intros i p kind Hi Hp Hk. rewrite Hrd in Hi.
      destruct (decisions_of_nth _ _ _ _ _ Hi) as (n & kind' & Hn & Hk' & Hd).
      rewrite Hk3, Hk' in Hk. injection Hk as <-.
      destruct (is_node_used sol n kind') eqn:Hu; [ discriminate Hd | ].
      unfold is_node_used in Hu. unfold is_var_used.
      destruct kind'; try discriminate Hu;
        (destruct (s_has_usage sol (Node_variable p)) eqn:E;
         [ | reflexivity ]);
        destruct (Hret0 i n (Node_variable p) Hn Hp) as [Hhu _];
        rewrite (Hhu E) in Hu; discriminate Hu. }
    rewrite Hm_ret.
    change (firstn (length fd) (rmask ++ repeat true (length fd)))
      with (pad_mask (length fd) rmask).
    rewrite Lfd.
    destruct (forallb (fun '(keep, (decision, _)) =>
                         Bool.eqb keep (is_kept decision))
                (combine (pad_mask (length rd) rmask) fd)) eqn:Hall.
    + (* No wrapper: the handler keeps what the code keeps. *)
      right. left. eexists.
      split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
      apply Hrel_callee. apply Ret_base. right. left. exists rmask.
      rewrite Hres0.
      split; [ exact Hret0 | ].
      split; [ unfold resolve; rewrite Hc0'; exact Hcrel0 | ].
      rewrite Lr, <- Lrd.
      unfold fd in Hall.
      assert (Lk : length rd = length (unarize (ap_return_arity ap)))
        by (rewrite Lrd, Hk3; reflexivity).
      exact (mask_of_forallb rd _ _ (length_pad_mask _ _) Lk Hall).
    + (* The wrapper, then the handler, which is not a return
         continuation of code: one keeps what the code keeps. *)
      right. right. right. eexists. exists 0. cbv zeta. split.
      * cbn [run]. rewrite step_let_cont_non_recursive. cbv beta iota.
        rewrite (Hstep' first_fresh).
        -- reflexivity.
        -- unfold resolve. cbn [lookup].
           apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity.
      * apply Hrel_callee. apply Ret_base. right. right. left.
        exists (ap_continuation ap), rmask, rd, (unarize (ap_return_arity ap)),
          vars', conts', c0'.
        split; [ | split; [ exact Hc0' | ] ].
        { unfold resolve, cc_wrapper. cbn [lookup]. rewrite Pos.eqb_refl.
          reflexivity. }
        split; [ rewrite Hres0; exact Hret0 | ].
        split; [ rewrite Hres0; exact Hcrel0 | ].
        split; [ reflexivity | ].
        split; [ rewrite Hk3; reflexivity | ].
        split; [ rewrite Lrd, Hk3; reflexivity | ].
        split; [ exact Hkeep | ].
        rewrite Hres0. exact Hdel.
  - (* The return continuation of the code the call is in: directly if
       it keeps what the code called keeps, at the nodes of the code
       called, to which the nodes of the call transport what they have;
       otherwise through the wrapper, over the return continuation. *)
    destruct (proj2 (Hnr_ap _ _ HKr) eq_refl) as (Hcount & HK).
    unfold count_is in Hcount. rewrite Hns0 in Hcount.
    apply Nat.eqb_eq in Hcount.
    pose proof (Hret_ce _ Hns0) as Hrn. subst returns.
    rewrite <- HK, Hk3 in Hrr0. rewrite <- HK, Hk3 in Hlm0.
    assert (Lm0 : length m0 = length rd) by (rewrite Hlm0, Lrd; reflexivity).
    assert (Lns0 : length (cd_return cd) = length ns0)
      by (rewrite Lr, Hcount, Hk3; reflexivity).
    assert (Lk : length rd = length (unarize (ap_return_arity ap)))
      by (rewrite Lrd, Hk3; reflexivity).
    destruct Hm0 as [Hm0 | [Hm0 Hall0]].
    2: {
    rewrite Hm0, Lfd.
    change (firstn (length rd) ([] ++ repeat true (length rd)))
      with (pad_mask (length rd) []).
    replace (pad_mask (length rd) []) with (pad_mask (length rd) m0)
      by (rewrite pad_mask_nil, <- Lm0, pad_mask_self;
          exact (forallb_true_repeat m0 Hall0)).
    destruct (forallb (fun '(keep, (decision, _)) =>
                         Bool.eqb keep (is_kept decision))
                (combine (pad_mask (length rd) m0) fd)) eqn:Hall.
    + unfold fd in Hall.
      pose proof (mask_of_forallb rd _ _ (length_pad_mask _ _) Lk Hall)
        as Hm0'.
      rewrite <- Lm0, pad_mask_self in Hm0'.
      right. left. eexists.
      split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
      apply Hrel_callee. rewrite Hm0'.
      apply (ret_rel_transport _ _ _ _ ns0); [ exact Hrr0 | | ].
      * exact Lns0.
      * intros i n n' Hn Hn'. exact (Htr_ret i n n' Hn Hn').
    + right. right. right. eexists. exists 0. cbv zeta. split.
      * cbn [run]. rewrite step_let_cont_non_recursive. cbv beta iota.
        rewrite (Hstep' first_fresh).
        -- reflexivity.
        -- unfold resolve. cbn [lookup].
           apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity.
      * apply Hrel_callee.
        apply (ret_wrapper_ret_intro _ _ m0 _ _ ns0 _ _ (ap_continuation ap)
                 rd vars' conts').
        { unfold resolve, cc_wrapper, fd. cbn [lookup].
          rewrite Pos.eqb_refl, Hk3. reflexivity. }
        { exact Hrr0. }
        { exact Lrd. }
        { exact (eq_sym Lns0). }
        { rewrite Lr. symmetry. exact Lrd. }
        { unfold fd in Hall. rewrite <- Lm0, pad_mask_self in Hall.
          apply (mask_mismatch rd m0 (unarize (ap_return_arity ap)) Lm0 Lk
                   Hall). }
        { intros i n n' Hn Hn'. exact (Htr_ret i n' n Hn' Hn). }
        { rewrite <- Hk3. exact Hkeep. }
        { intros i Hi. destruct (Hdel0 i Hi) as (n & Hn & Hnu & Hkind).
          split; [ | rewrite <- Hk3; exact Hkind ].
          intros n' Hn' Hu'. apply Hnu.
          exact (proj1 (Htr_ret i n' n Hn' Hn) Hu'). } }
    rewrite Hm0, Lfd.
    change (firstn (length rd) (m0 ++ repeat true (length rd)))
      with (pad_mask (length rd) m0).
    destruct (forallb (fun '(keep, (decision, _)) =>
                         Bool.eqb keep (is_kept decision))
                (combine (pad_mask (length rd) m0) fd)) eqn:Hall.
    + unfold fd in Hall.
      pose proof (mask_of_forallb rd _ _ (length_pad_mask _ _) Lk Hall)
        as Hm0'.
      rewrite <- Lm0, pad_mask_self in Hm0'.
      right. left. eexists.
      split; [ exact (Hstep' (ap_continuation ap) conts' eq_refl) | ].
      apply Hrel_callee. rewrite Hm0'.
      apply (ret_rel_transport _ _ _ _ ns0); [ exact Hrr0 | | ].
      * exact Lns0.
      * intros i n n' Hn Hn'. exact (Htr_ret i n n' Hn Hn').
    + right. right. right. eexists. exists 0. cbv zeta. split.
      * cbn [run]. rewrite step_let_cont_non_recursive. cbv beta iota.
        rewrite (Hstep' first_fresh).
        -- reflexivity.
        -- unfold resolve. cbn [lookup].
           apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity.
      * apply Hrel_callee.
        apply (ret_wrapper_ret_intro _ _ m0 _ _ ns0 _ _ (ap_continuation ap)
                 rd vars' conts').
        { unfold resolve, cc_wrapper, fd. cbn [lookup].
          rewrite Pos.eqb_refl, Hk3. reflexivity. }
        { exact Hrr0. }
        { exact Lrd. }
        { exact (eq_sym Lns0). }
        { rewrite Lr. symmetry. exact Lrd. }
        { unfold fd in Hall. rewrite <- Lm0, pad_mask_self in Hall.
          apply (mask_mismatch rd m0 (unarize (ap_return_arity ap)) Lm0 Lk
                   Hall). }
        { intros i n n' Hn Hn'. exact (Htr_ret i n' n Hn' Hn). }
        { rewrite <- Hk3. exact Hkeep. }
        { intros i Hi. destruct (Hdel0 i Hi) as (n & Hn & Hnu & Hkind).
          split; [ | rewrite <- Hk3; exact Hkind ].
          intros n' Hn' Hu'. apply Hnu.
          exact (proj1 (Htr_ret i n' n Hn' Hn) Hu'). }
Qed.

(* A direct call: the first run enters the body of the code of its
   callee, and the second the rebuilt body, with the parameters kept
   related, and [my_closure]; the continuations of the code are those of
   the call, or the wrapper that the rebuild gives it. *)
Lemma direct_answered (ap : apply_expr) (id : code_id) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (rcd rcd' : read_record)
    (e' : expr) :
  ap_call_kind ap = Direct id ->
  related (State (Apply ap) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Apply ap) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  intros Hkind. related_intro.
  destruct He as (Hfrag_ap & Hnoted_callee & Hnoted_args & returns & exn
                  & Hret_ce & Hexn_ce & Hcall).
  destruct Hok as (Hlive & (bound & [Hb_sub Hb_rets] & Hexits) & _ & Hcf
                   & (Hfresh & (ks & Hks & Hagree) & xs & Hxs & Hxagree)
                   & Hnr_ap).
  cbn [exits_bound] in Hexits. cbn [checked] in Hcf.
  cbn [calls_kinded] in Hks. cbn [raises_kept] in Hxs.
  apply andb_true_iff in Hexits as [Hk_exn Hk_ret].
  apply orb_true_iff in Hk_exn, Hk_ret.
  assert (Hk_exn' : among (ap_exn_continuation ap) (map fst (env_conts en))
                    || count_is ce (ap_exn_continuation ap)
                         (S (length (ap_exn_extra_args ap)))
                    = true).
  { apply orb_true_iff. destruct Hk_exn as [H | H];
      [ left; exact (Hb_sub _ H) | right; exact H ]. }
  assert (Hk_ret' : among (ap_continuation ap) (map fst (env_conts en))
                    || count_is ce (ap_continuation ap)
                         (length (unarize (ap_return_arity ap)))
                    = true).
  { apply orb_true_iff. destruct Hk_ret as [H | H];
      [ left; exact (Hb_sub _ H) | right; exact H ]. }
  (* The return continuations of code that changes: counted, and not
     raised to. *)
  assert (Hcount_rets : ∀ (r : continuation) (K : list flambda_kind),
            lookup r rets = Some K -> ap_continuation ap = r ->
            count_is ce r (length (unarize (ap_return_arity ap))) = true).
  { intros r K Hr E. subst r. destruct Hk_ret as [H | H]; [ | exact H ].
    rewrite (Hb_rets _ K Hr) in H. discriminate H. }
  assert (Hexn_rets : ∀ (r : continuation) (K : list flambda_kind),
            lookup r rets = Some K -> ap_exn_continuation ap <> r).
  { intros r K Hr E. destruct (Hnr_ap r K Hr) as [H _].
    cbn [no_raise_to] in H. rewrite E, Pos.eqb_refl in H. discriminate H. }
  clear Hk_exn Hk_ret. rename Hk_exn' into Hk_exn. rename Hk_ret' into Hk_ret.
  assert (Hff_exn : ap_exn_continuation ap <> first_fresh).
  { intros E. apply not_true_iff_false in Hfresh. apply Hfresh.
    apply among_In. cbn [continuations_of continuations_in_apply].
    right. left. exact E. }
  apply negb_true_iff in Hcf as Hexf0.
  unfold call_edges in Hcall. rewrite Hkind in Hcall.
  destruct Hcall as (witness & apply_node & called_node & Hargs & Hretacc
                     & Hexnacc & Hcalled & Hused & Hcond & Happly).
  pose proof Hflow as (ctx0 & ce0 & _ & _ & _ & _ & _ & _ & Hcodes & _).
  pose proof (call_step is_local G noted deps cds keep codes symbols False ap
      vars
                conts statics hp rcd id Hkind Hfrag_ap Hcodes) as Hcs.
  unfold answered. revert Hflow' Hcs.
  destruct (step (State (Apply ap) vars conts statics hp rcd))
    as [s' | k vs0 hp0 rcd0 | ]; [ | intros _ [] | intros; exact I ].
  intros Hflow' (callee & vs & extra & slots & c & callee_vars & Hcallee
                 & Hvs & Hextra & Hclosure & Hcode & Hk1 & Hk2 & Hk3 & Hbind
                 & ->).
  (* The extra arguments of the exception, in the second run. *)
  destruct (rewrite_simples_kinds en vars vars' Henv _ extra Hextra)
    as (extra' & Hextra' & Hkx).
  (* The rebuilt call. *)
  assert (Hdead : (simple_is_dead sol en (ap_callee ap)
                   || existsb (simple_is_dead sol en) (ap_args ap)) = false).
  { apply orb_false_iff. split.
    - exact (simple_not_dead en vars vars' _ _ Hvars Hnoted_callee Hcallee).
    - apply not_true_iff_false. intros Hex.
      apply existsb_exists in Hex as (s & Hin & Hd).
      destruct (eval_simples_member vars _ vs s Hvs Hin) as (w & Hw).
      rewrite (simple_not_dead en vars vars' s w Hvars (Hnoted_args s Hin) Hw)
        in Hd.
      discriminate Hd. }
  (* What the call reads. *)
  pose proof (au_hu _ _ _ Hc _ (any_usage_input _ _ _ Hc _ Hused))
    as Hhu_called.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_called Hcalled) as Hhu_w.
  pose proof (cond_alias_alias is_local G sol Hc ctx witness apply_node Hlive
                Hcond) as Hal_wa.
  pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_w Hal_wa) as Hhu_a.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_a Happly) as Hhu_callee.
  destruct (rewrite_simple_eval en vars vars' (ap_callee ap) callee Henv
              Hnoted_callee Hcallee)
    as (callee' & Hcallee' & _ & Hkept_c).
  destruct (Hkept_c (kept_of_hu en _ Hhu_callee)) as [Hrw_c Hrel_c].
  rewrite Hrw_c in Hcallee'.
  destruct (rel_closure_alloc statics callee callee' _ id slots Hsflow Hrel_c
              Hclosure)
    as (b & Hb & Hrel_b & Hcalled_b).
  destruct (rel_closure_of statics statics' callee callee' _ id slots Hst
              Hrel_c Hhu_callee Hclosure)
    as (slots' & Hclosure').
  destruct (code_facts is_local G noted deps cds keep codes symbols False Hnd
              statics id c Hcodes Hcode)
    as (Hin_codes & cd & Hcd & _ & _ & Lp & Lr).
  destruct (Hcds id cd Hcd) as (Hpar & Hretc & Hexnc & Hcodeid & Hasid).
  pose proof (called_alloc is_local G sol cds Hc Hcds b id Hcalled_b) as Hbb.
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & Hcw & _).
  set (w := cd_known_arity_call_witness cd) in *.
  assert (Hw_self : may_come_from sol w w).
  { apply (alloc_self is_local G sol Hc _ _ _ Hcodeid).
    exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hasid)). }
  assert (Hw : may_come_from sol w witness).
  { apply (may_come_from_alias _ _ _ Hc _ _ apply_node Hal_wa).
    exact (may_come_from_read _ _ _ Hc w b _ apply_node w _
             call_witness_not_local Hcw Happly Hb Hw_self). }
  assert (Hlive' : au (Node_code_id id)).
  { refine (read_any_usage _ _ _ Hc w witness called_node _ _ _ Hcodeid
              Hcalled Hw (any_usage_input _ _ _ Hc _ Hused)).
    apply (not_local is_local Hlocal); intros; discriminate. }
  assert (Hcode' : lookup id (static_code statics') = Some (rebuild_c id c)).
  { pose proof Hst as (_ & _ & _ & Hsc). destruct (Hsc id) as [H | [_ H]].
    - rewrite H, Hcode. reflexivity.
    - exfalso. exact (H (au_hu _ _ _ Hc _ Hlive')). }
  subst e'. cbn [rebuild_expr_cc] in Hconts |- *.
  (* The code is of the fragment, and as the unit asks. *)
  pose proof Hfrag_ap as Hall. unfold apply_in_flow_fragment in Hall.
  rewrite Hkind, forallb_forall in Hall. specialize (Hall (id, c) Hin_codes).
  simpl in Hall. rewrite Pos.eqb_refl in Hall. simpl in Hall.
  apply andb_true_iff in Hall as [Htupled Hfull].
  apply negb_true_iff in Htupled. apply Nat.eqb_eq in Hfull.
  destruct (Hcode_checks id c Hin_codes)
    as (Hscr_c & Hcf_c & Hraises_c & Hclosed_c & Hkinded_c & Hfresh_c).
  pose proof (Hcodes id c Hcode) as [_ Hedges_c].
  pose proof Hk2 as Hk2b.
  apply kinds_eqb_eq in Hk1. apply kinds_per_parameter_eqb_eq in Hk2.
  apply unarize_per_parameter_eq in Hk2. pose proof Hk3 as Hk3b.
  apply arity_eqb_eq in Hk3.
  assert (Hnc : ∀ c0 : const, ap_callee ap <> Const c0).
  { intros c0 E. rewrite E in Hcallee. injection Hcallee as <-.
    discriminate Hclosure. }
  pose proof (rewrite_call_kind_cases ap id b w apply_node Hkind Happly Hb Hcw
                Hcodeid Hw_self) as Hkind'.
  assert (Hl_exn : is_local (Field_return_of_call Exn) = false)
    by (apply (not_local is_local Hlocal); intros; discriminate).
  assert (Hl_ret : ∀ i : nat,
            is_local (Field_return_of_call (Normal i)) = false)
    by (intros i; apply (not_local is_local Hlocal); intros; discriminate).
  assert (Htr_exn : transports (cd_exn cd) exn)
    by exact (read_transports _ _ w witness _ Hl_exn Hexnc Hexnacc Hw).
  assert (Htr_ret :
    ∀ (i : nat) (n n' : node),
      nth_error returns i = Some n -> nth_error (cd_return cd) i = Some n' ->
      transports n' n).
  { intros i n n' Hn Hn'.
    exact (read_transports _ _ w witness _ (Hl_ret i) (Hretc i n' Hn')
             (Hretacc i n Hn) Hw). }
  destruct (code_calling_convention_change sol cannot_change code_deps
              unit_code_ids id (metadata_of_code c)) as [ | pd rd ] eqn:Hun.
  2: {
    (* The code changes its calling convention: the call is known to call
       it, and gives it the arguments it keeps. *)
    assert (Hcc_id : cannot_change (Node_code_id id) = false).
    { unfold code_calling_convention_change,
        cannot_change_calling_convention in Hun.
      destruct (cannot_change (Node_code_id id));
        [ rewrite orb_true_r in Hun; discriminate Hun | reflexivity ]. }
    pose proof (direct_call_recognized ap id b w apply_node witness
                  called_node Hkind Hnc Happly Hal_wa Hcalled
                  (any_usage_input _ _ _ Hc _ Hused) Hb Hcw Hcodeid Hw_self
                  Hcc_id) as Hcalled_ap.
    pose proof (Hmetadatas id c Hin_codes Hcc_id) as Hmeta.
    assert (Hrw : rebuild_apply_cc F sol cannot_change code_deps
                    preserve_direct_calls unit_code_ids code_metadatas
                    first_fresh en ap
                  = make_apply_wrapper_cc first_fresh en (cc_call en ap pd rd)
                      (combine rd (unarize (ap_return_arity ap)))).
    { unfold rebuild_apply_cc, apply_calling_convention_change.
      rewrite Hcalled_ap. unfold get_calling_convention_change.
      rewrite Hmeta, Hun, Hdead. reflexivity. }
    rewrite Hrw in Hconts |- *.
    assert (Hacc : apply_calling_convention_change F sol cannot_change
                     code_deps unit_code_ids code_metadatas ap
                   = Changing_calling_convention pd rd).
    { unfold apply_calling_convention_change. rewrite Hcalled_ap.
      unfold get_calling_convention_change. rewrite Hmeta, Hun.
      reflexivity. }
    assert (Hnr_cc :
      ∀ (r : continuation) (K : list flambda_kind),
        lookup r rets = Some K ->
        ap_exn_continuation ap <> r
        ∧ (ap_continuation ap = r ->
           count_is ce r (length (unarize (ap_return_arity ap))) = true
           ∧ unarize (ap_return_arity ap) = K)).
    { intros r K Hr. split; [ exact (Hexn_rets r K Hr) | ].
      intros E. split; [ exact (Hcount_rets r K Hr E) | ].
      destruct (Hnr_ap r K Hr) as [_ H].
      cbn [ret_calls_ok] in H.
      rewrite E, Pos.eqb_refl in H. cbn [negb orb] in H.
      apply kinds_eqb_eq in H. symmetry. exact H. }
    assert (Hpdrd :
      pd = decisions_of sol (cd_params cd)
             (unarize (cm_params_arity (metadata_of_code c)))
      ∧ rd = decisions_of sol (cd_return cd)
               (unarize (cm_result_arity (metadata_of_code c)))).
    { unfold code_calling_convention_change,
        cannot_change_calling_convention in Hun.
      rewrite Hcc_id, orb_false_r in Hun.
      destruct (negb (among id unit_code_ids)); [ discriminate Hun | ].
      rewrite (Hcode_deps id Hcc_id), Hcd in Hun.
      injection Hun as <- <-. split; reflexivity. }
    destruct (rewrite_simples_eval en vars vars' Henv (ap_args ap) vs
                Hnoted_args Hvs)
      as (vs' & Hvs' & Hkinds' & Hrels').
    assert (Hlen' : length vs' = length vs).
    { rewrite <- (length_map kind_of_value vs'), Hkinds', length_map.
      reflexivity. }
    assert (La : length (ap_args ap) = length (unarize (ap_args_arity ap))).
    { rewrite <- Hk1, length_map. symmetry.
      exact (eval_simples_length _ _ _ Hvs). }
    pose proof Hk2b as Hk2'. apply kinds_per_parameter_eqb_eq in Hk2'.
    set (mask := arguments_used_by_known_arity_call F sol
                   (node_of_simple (ap_callee ap)) (ap_args ap)) in *.
    destruct c as [params mc ret exn_k md cbody].
    cbn [code_body code_params code_my_closure code_return_continuation
         code_exn_continuation code_params_arity code_result_arity
         code_is_tupled metadata_of_code] in *.
    destruct Hedges_c as (cd' & Hcd' & _ & _ & _ & _ & _ & Hmc & Hps & Hne
                          & Hbinds & Hcounted & Hbody).
    rewrite (lookup_nodup_in _ _ _ Hnd Hcd') in Hcd. injection Hcd as Hcd.
    subst cd'.
    destruct Hpdrd as [Hpd Hrd].
    destruct (Hcode_rets id params mc ret exn_k md cbody Hexf0 Hin_codes)
      as [Hraise_c Hrc_c].
    assert (Hrel_mc : rel callee (Node_variable mc) callee').
    { apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _
               _ _ (alias_input _ _ _ Hc _ _ Hmc)).
      apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _
               _ _ (alias_propagate _ _ _ Hc _ _ _ Hlive' Hprop)).
      exact Hrel_b. }
    assert (Hkind_c : kind_of_value callee = Value).
    { destruct callee; try discriminate Hclosure; reflexivity. }
    assert (Henv0 : env_ok [(mc, Value)] [(mc, callee)] [(mc, callee')]).
    { intros x v Hx. simpl in Hx |- *.
      destruct (Pos.eqb_spec x mc) as [-> | _]; [ | discriminate Hx ].
      injection Hx as <-. rewrite Hkind_c. split; [ reflexivity | ].
      intros _. exists callee'. split; [ reflexivity | exact Hrel_mc ]. }
    assert (Lvs : length params = length vs)
      by exact (eq_sym (bind_params_length _ _ _ _ Hbind)).
    assert (Lpd : length (map is_kept pd) = length params).
    { rewrite length_map, Hpd, length_decisions_of, Lp, Nat.min_id, <- Hk2,
        <- Hk1, length_map.
      symmetry. exact Lvs. }
    assert (Hrel_params :
      ∀ (i : nat) (p : variable) (v v' : value),
        nth_error params i = Some p -> nth_error vs i = Some v ->
        nth_error vs' i = Some v' ->
        is_var_used sol p (kind_of_value v) = true ->
        nth_error (map is_kept pd) i = Some true
        ∧ rel v (Node_variable p) v').
    { intros i p v v' Hp Hv Hv' Hu.
      destruct (argument_kind_ok (kind_of_value v)) eqn:Hkv.
      2: { (* The poison of [Region] or [Rec_info]: the code keeps the
              parameter, and the argument is the same in the two runs. *)
           destruct (nth_error (cd_params cd) i) as [q | ] eqn:Hq.
           2: { exfalso. apply nth_error_None in Hq.
                assert (i < length params)
                  by (apply nth_error_Some; congruence).
                rewrite Lp, <- Hk2, <- Hk1, length_map, <- Lvs in Hq. lia. }
           split;
             [ | exact (rel_region_argument vs vs' v v' i _ Hkv Hkinds' Hv
                          Hv') ].
           assert (Hkind_i : nth_error (unarize md.(cm_params_arity)) i
                             = Some (kind_of_value v)).
           { rewrite <- Hk2, <- Hk1, nth_error_map, Hv. reflexivity. }
           assert (Hi : i < length pd).
           { rewrite Hpd, length_decisions_of, Lp, Nat.min_id.
             apply nth_error_Some. congruence. }
           destruct (nth_error pd i) as [d | ] eqn:Hd;
             [ | apply nth_error_None in Hd; lia ].
           rewrite nth_error_map, Hd. cbn [option_map]. f_equal.
           rewrite Hpd in Hd.
           destruct (decisions_of_nth _ _ _ _ _ Hd)
             as (n & kind & Hn & Hk' & ->).
           rewrite Hkind_i in Hk'. injection Hk' as <-.
           unfold is_node_used.
           destruct (kind_of_value v); simpl in Hkv; try discriminate Hkv;
             reflexivity. }
      assert (Hhu_p : hu (Node_variable p)).
      { destruct (kind_of_value v); simpl in Hkv, Hu;
          try discriminate Hkv; exact Hu. }
      destruct (nth_error (cd_params cd) i) as [q | ] eqn:Hq.
      2: { exfalso. apply nth_error_None in Hq.
           assert (i < length params) by (apply nth_error_Some; congruence).
           rewrite Lp, <- Hk2, <- Hk1, length_map, <- Lvs in Hq. lia. }
      pose proof (alias_input _ _ _ Hc _ _
                    (Hps p q (nth_error_combine _ _ _ _ _ Hp Hq))) as Hal_pq.
      pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_p Hal_pq) as Hhu_q.
      split.
      - assert (Hkind_i : nth_error (unarize md.(cm_params_arity)) i
                          = Some (kind_of_value v)).
        { rewrite <- Hk2, <- Hk1, nth_error_map, Hv. reflexivity. }
        assert (Hi : i < length pd).
        { rewrite Hpd, length_decisions_of, Lp, Nat.min_id.
          apply nth_error_Some. congruence. }
        destruct (nth_error pd i) as [d | ] eqn:Hd;
          [ | apply nth_error_None in Hd; lia ].
        rewrite nth_error_map, Hd. cbn [option_map]. f_equal.
        rewrite Hpd in Hd.
        destruct (decisions_of_nth _ _ _ _ _ Hd)
          as (n & kind & Hn & Hk' & ->).
        rewrite Hq in Hn. injection Hn as <-.
        rewrite Hkind_i in Hk'. injection Hk' as <-.
        unfold is_node_used.
        destruct (kind_of_value v); simpl in Hkv; try discriminate Hkv;
          rewrite Hhu_q; reflexivity.
      - destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
        assert (Hm : nth_error mask i = Some true).
        { apply (known_mask_used _ b w q _ i Hb Hbb Hcw Hw_self
                   (Hpar i q Hq) Hhu_q).
          apply nth_error_Some. congruence. }
        pose proof (Hrels' i s v v' Hs Hv Hv'
                      (kept_of_hu en _
                         (known_mask_kept _ witness apply_node _ i s Happly
                            Hal_wa (Hargs i s Hs) Hm))) as Hrel_s.
        pose proof (flows_hs is_local G sol keep called no_loc Hc _ _
                      (simple_flows is_local G sol noted cds keep no_loc Hc
                          Hconsts
                         Hsyms _ _ _ Hvars
                         (Hnoted_args s (nth_error_In _ _ Hs)) Hsv))
          as Hhs.
        apply (rel_alias G sol keep called (exact = true) True is_local Hc _
                 _ _ _ Hal_pq).
        exact (rel_argument w witness q (node_of_simple s) i v v' Hw
                 (Hargs i s Hs) (Hpar i q Hq) Hhs Hrel_s). }
    destruct (bind_params_rel params vs vs' (map is_kept pd) [(mc, Value)]
                [(mc, callee)] [(mc, callee')] callee_vars Henv0 Hlen' Lpd
                Hrel_params Hbind)
      as (callee_vars' & Hbind' & Hok').
    subst hp'.
    exact (call_entered_cc ap vars vars' conts conts' statics statics' hp rcd
      rcd' en
             ce returns exn ks xs id params mc ret exn_k md cbody cd pd rd
             callee callee' slots' vs vs' extra extra' callee_vars
             callee_vars' rets Hun Hret_ce Hexn_ce Henv Hconts Hnr_cc Hst
             Hk_exn Hk_ret Hks Hagree Hxs Hxagree Hff_exn Hextra Hextra' Hkx
             Hflow' Hcallee' Hclosure' Hkind' Hcode' Hscr_c Hcf_c Hraises_c
             Hclosed_c Hkinded_c Hfresh_c (Hraise_c Hne) Hrc_c Hne Hbinds
             Hcounted
             Hbody Lr Lp Hpd Hrd Hvs' Hkinds' Hk1 Hk2' Hk3 La Hbind Hbind' Hok'
             Hlive' Htr_exn Htr_ret). }
  (* The code keeps its calling convention, and the call is rebuilt as in
     FlambdaReaperRebuild.v. *)
  rewrite (rebuild_apply_unchanged_code ap id (metadata_of_code c) en b w
             apply_node Hkind Happly Hb Hcw Hcodeid Hw_self Hun)
    in Hconts |- *.
  pose proof (apply_cc_unchanged_code ap id (metadata_of_code c) b w
                apply_node Hkind Happly Hb Hcw Hcodeid Hw_self Hun) as Hacc0.
  assert (Hnr_plain :
    ∀ (r : continuation) (K : list flambda_kind),
      lookup r rets = Some K ->
      ap_exn_continuation ap <> r
      ∧ (ap_continuation ap = r ->
         count_is ce r (length (unarize (ap_return_arity ap))) = true
         ∧ unarize (ap_return_arity ap) = K)).
  { intros r K Hr. split; [ exact (Hexn_rets r K Hr) | ].
    intros E. split; [ exact (Hcount_rets r K Hr E) | ].
    destruct (Hnr_ap r K Hr) as [_ H].
    cbn [ret_calls_ok] in H.
    rewrite E, Pos.eqb_refl in H. cbn [negb orb] in H.
    apply kinds_eqb_eq in H. symmetry. exact H. }
  unfold rebuild_apply in Hconts |- *. rewrite Hdead in Hconts |- *.
  (* The arguments of the second run. *)
  assert (Hargs_eq :
    rebuild_arguments F sol en ap
    = mask_arguments en
        (arguments_used_by_known_arity_call F sol
           (node_of_simple (ap_callee ap)) (ap_args ap))
        (ap_args ap)).
  { unfold rebuild_arguments. rewrite Hkind.
    destruct (ap_callee ap) as [x | sym | c0];
      [ reflexivity | reflexivity | destruct (Hnc c0 eq_refl) ]. }
  set (mask := arguments_used_by_known_arity_call F sol
                 (node_of_simple (ap_callee ap)) (ap_args ap)) in *.
  assert (Hmlen : length mask = length (ap_args ap)).
  { destruct (known_mask_cases (node_of_simple (ap_callee ap)) (ap_args ap))
      as [[Hm _] | (ws & Hm & _)]; fold mask in Hm; rewrite Hm;
      [ apply repeat_length | rewrite length_map; apply length_seq ]. }
  destruct (mask_arguments_eval en vars vars' Henv mask (ap_args ap) vs
              Hnoted_args Hvs Hmlen)
    as (vs' & Hvs' & Hkinds' & Hrels').
  { intros i s Hs Hm.
    exact (known_mask_kept _ witness apply_node _ i s Happly Hal_wa
             (Hargs i s Hs) Hm). }
  assert (Hlen' : length vs' = length vs).
  { rewrite <- (length_map kind_of_value vs'), Hkinds', length_map.
    reflexivity. }
  (* The callee of the second run. *)
  destruct c as [params mc ret exn_k md cbody].
  cbn [code_body code_params code_my_closure code_return_continuation
       code_exn_continuation code_params_arity code_result_arity
       code_is_tupled] in *.
  destruct Hedges_c as (cd' & Hcd' & _ & _ & _ & _ & _ & Hmc & Hps & Hne
                        & Hbinds & Hcounted & Hbody).
  rewrite (lookup_nodup_in _ _ _ Hnd Hcd') in Hcd. injection Hcd as Hcd.
  subst cd'.
  (* [my_closure] and the parameters. *)
  assert (Hrel_mc : rel callee (Node_variable mc) callee').
  { apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _ _ _
             (alias_input _ _ _ Hc _ _ Hmc)).
    apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _ _ _
             (alias_propagate _ _ _ Hc _ _ _ Hlive' Hprop)).
    exact Hrel_b. }
  assert (Hkind_c : kind_of_value callee = Value).
  { destruct callee; try discriminate Hclosure; reflexivity. }
  assert (Henv0 : env_ok [(mc, Value)] [(mc, callee)] [(mc, callee')]).
  { intros x v Hx. simpl in Hx |- *.
    destruct (Pos.eqb_spec x mc) as [-> | _]; [ | discriminate Hx ].
    injection Hx as <-. rewrite Hkind_c. split; [ reflexivity | ].
    intros _. exists callee'. split; [ reflexivity | exact Hrel_mc ]. }
  assert (Hrel_params :
    ∀ (i : nat) (p : variable) (v v' : value),
      nth_error params i = Some p -> nth_error vs i = Some v ->
      nth_error vs' i = Some v' ->
      is_var_used sol p (kind_of_value v) = true ->
      nth_error (repeat true (length params)) i = Some true
      ∧ rel v (Node_variable p) v').
  { intros i p v v' Hp Hv Hv' Hu.
    pose proof (bind_params_length _ _ _ _ Hbind) as Lvs.
    assert (Hi : i < length params)
      by (apply nth_error_Some; congruence).
    split; [ exact (nth_error_repeat_lt true _ _ Hi) | ].
    destruct (argument_kind_ok (kind_of_value v)) eqn:Hkv.
    2: { (* The poison of [Region] or [Rec_info], the same in the two
            runs. *)
         exact (rel_region_argument vs vs' v v' i _ Hkv Hkinds' Hv Hv'). }
    assert (Hhu_p : hu (Node_variable p)).
    { destruct (kind_of_value v); simpl in Hkv, Hu;
        try discriminate Hkv; exact Hu. }
    destruct (nth_error (cd_params cd) i) as [q | ] eqn:Hq.
    2: { exfalso. apply nth_error_None in Hq.
         rewrite Lp, <- Hk2, <- Hk1, length_map, Lvs in Hq. lia. }
    pose proof (alias_input _ _ _ Hc _ _
                  (Hps p q (nth_error_combine _ _ _ _ _ Hp Hq))) as Hal_pq.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_p Hal_pq) as Hhu_q.
    destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
    assert (Hm : nth_error mask i = Some true).
    { apply (known_mask_used _ b w q _ i Hb Hbb Hcw Hw_self (Hpar i q Hq)
               Hhu_q).
      apply nth_error_Some. congruence. }
    pose proof (Hrels' i s v v' Hs Hv Hv' Hm) as Hrel_s.
    pose proof (flows_hs is_local G sol keep called no_loc Hc _ _
                  (simple_flows is_local G sol noted cds keep no_loc Hc Hconsts
                      Hsyms
                     _ _ _ Hvars (Hnoted_args s (nth_error_In _ _ Hs)) Hsv))
      as Hhs.
    apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _ _ _
             Hal_pq).
    exact (rel_argument w witness q (node_of_simple s) i v v' Hw
             (Hargs i s Hs) (Hpar i q Hq) Hhs Hrel_s). }
  destruct (bind_params_rel params vs vs' (repeat true (length params))
              [(mc, Value)] [(mc, callee)] [(mc, callee')] callee_vars Henv0
              Hlen' (repeat_length _ _) Hrel_params Hbind)
    as (callee_vars' & Hbind' & Hok').
  rewrite !filter_kept_all in Hbind'.
  subst hp'.
  exact (call_entered ap vars vars' conts conts' statics statics' hp rcd rcd' en
    ce
           returns exn ks xs id params mc ret exn_k md cbody cd callee callee'
           slots' vs vs' extra extra' callee_vars callee_vars' rets Hun
           Hret_ce Hexn_ce Henv Hconts Hnr_plain Hst Hk_exn Hk_ret Hks Hagree
           Hxs Hxagree
           Hff_exn Hextra Hextra' Hkx Hflow' Hcallee' Hclosure' Hkind' Hcode'
           Htupled Hfull Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c Hfresh_c
           Hne Hbinds Hcounted Hbody Lr
           (eq_trans (f_equal (eval_simples vars') Hargs_eq) Hvs') Hkinds' Hk1
           Hk2 Hk2b Hk3 Hk3b Hbind Hbind' Hok' Hlive' Htr_exn Htr_ret
           (proj1 (Hcode_rets id params mc ret exn_k md cbody Hexf0
                     Hin_codes) Hne)
           (proj2 (Hcode_rets id params mc ret exn_k md cbody Hexf0
                     Hin_codes))).
Qed.

(* ------------------------------------------------------------------------ *)
(* Returns, and calls of unknown arity of any shape                         *)
(* ------------------------------------------------------------------------ *)

(* What a step of the first run comes to keeps the invariant of the flow. *)
Definition result_flow (r : step_result) : Prop :=
  match r with
  | Step s' =>
    flow_invariant is_local G sol noted deps cds keep codes symbols u False
    no_loc s'
  | _ => True
  end.

(* A step of the first run, to [r], answered by the second, to [r']: an
   exit by an exit through the same continuation, with related values;
   a step by a step, or by more, to a related state. *)
Definition result_rel (r r' : step_result) : Prop :=
  match r with
  | Undefined => True
  | Exit k vs hp _ =>
    ∃ (vs' : list value) (rcd' : read_record),
      r' = Exit k vs' hp rcd' ∧ exits_related vs vs'
  | Step s' =>
    step s' = Undefined
    ∨ (∃ t' : state, r' = Step t' ∧ related s' t')
    ∨ ∃ (t1 t' : state) (n : nat),
        r' = Step t1 ∧ run (S n) t1 = Step t' ∧ related s' t'
  end.

Lemma run_succ_step (n : nat) (t t1 : state) :
  step t = Step t1 -> run (S n) t = run n t1.
Proof. intros H. cbn [run]. rewrite H. reflexivity. Qed.

Lemma answered_of_result (s t : state) :
  result_rel (step s) (step t) -> answered related exits_related measure s t.
Proof.
  unfold answered, result_rel.
  destruct (step s) as [s' | k vs hp rcd | ];
    [ | intros H; exact H | intros _; exact I ].
  intros [H | [(t' & Ht & Hr) | (t1 & t' & n & Ht & Hrun & Hr)]].
  - left. exact H.
  - right. left. exists t'. split; assumption.
  - right. right. right. exists t', n.
    rewrite (run_succ_step _ _ _ Ht). split; assumption.
Qed.

(* The same, after a step of the second run, where the first does not
   exit. *)
Lemma answered_of_result_after (s t t0 : state) :
  step t = Step t0 ->
  (∀ (k : continuation) (vs : list value) (hp : heap) (rcd : read_record),
     step s <> Exit k vs hp rcd) ->
  result_rel (step s) (step t0) ->
  answered related exits_related measure s t.
Proof.
  intros Ht0 Hnx. unfold answered, result_rel.
  destruct (step s) as [s' | k vs hp rcd | ];
    [ | destruct (Hnx k vs hp rcd eq_refl) | intros _; exact I ].
  intros [H | [(t' & Ht & Hr) | (t1 & t' & n & Ht & Hrun & Hr)]].
  - left. exact H.
  - right. right. right. exists t', 0.
    rewrite (run_succ_step _ _ _ Ht0), (run_succ_step _ _ _ Ht).
    split; [ reflexivity | exact Hr ].
  - right. right. right. exists t', (S n).
    rewrite (run_succ_step _ _ _ Ht0), (run_succ_step _ _ _ Ht).
    split; [ exact Hrun | exact Hr ].
Qed.

(* A step of the first run that is not an exit, answered by the second
   after one step more. *)
Lemma result_rel_after (r : step_result) (t1 : state) :
  (∀ (k : continuation) (vs : list value) (hp : heap) (rcd : read_record),
     r <> Exit k vs hp rcd) ->
  result_rel r (step t1) -> result_rel r (Step t1).
Proof.
  intros Hnx. unfold result_rel.
  destruct r as [s' | k vs hp rcd | ];
    [ | destruct (Hnx k vs hp rcd eq_refl) | intros _; exact I ].
  intros [H | [(t' & Ht & Hr) | (t2 & t' & n & Ht & Hrun & Hr)]].
  - left. exact H.
  - right. right. exists t1, t', 0. split; [ reflexivity | ].
    split; [ | exact Hr ]. rewrite (run_succ_step 0 _ _ Ht). reflexivity.
  - right. right. exists t1, t', (S n). split; [ reflexivity | ].
    split; [ | exact Hr ]. rewrite (run_succ_step _ _ _ Ht). exact Hrun.
Qed.

(* Values given at the nodes [ns], related where these have a use. *)
Definition at_nodes (ns : list node) (vs vs' : list value) : Prop :=
  length vs = length ns
  ∧ map kind_of_value vs' = map kind_of_value vs
  ∧ ∀ (i : nat) (n : node) (v v' : value),
      nth_error ns i = Some n -> nth_error vs i = Some v ->
      nth_error vs' i = Some v' ->
      hu n ∨ kind_of_value v = Region ∨ kind_of_value v = Rec_info ->
      rel v n v'.

(* Giving values to a way out [c] of the first run is answered by the
   second. *)
Definition returned_ok (c : cont_value) : Prop :=
  ∀ (c' : cont_value) (ns : list node) (vs vs' : list value)
    (statics statics' : static_env) (hp : heap) (rcd rcd' : read_record),
    way_out cont_rel over_rel ns c c' -> at_nodes ns vs vs' ->
    statics_rel statics statics' ->
    codes_ok is_local G noted deps cds keep codes symbols False statics ->
    statics_flow_ statics ->
    every (pa_ok statics) vs -> cont_pa_ok statics c ->
    result_flow (apply_cont_value c vs statics hp rcd) ->
    result_rel (apply_cont_value c vs statics hp rcd)
      (apply_cont_value c' vs' statics' hp rcd').

(* A call of unknown arity of what is at [N], with the arguments related
   at the witnesses that their groups reach from what [N] may come from,
   which the witnesses [ws] of the call stand for, returning to [rt]: the
   call in the first run is answered by the call in the second. *)
Definition called_ok (rt : cont_value) : Prop :=
  ∀ (N : node) (callee callee' : value) (vs vs' : list value)
    (a ret : arity) (rt' et et' : cont_value) (ws returns : list node)
    (exn : node) (statics statics' : static_env) (hp : heap)
    (rcd rcd' : read_record),
    exact = false -> indirect_codes -> hu N -> a <> [] ->
    length ws = length a ->
    rel callee N callee' -> pa_ok statics callee ->
    args_rel (from_node N) a vs vs' -> every (pa_ok statics) vs ->
    call_reads (from_node N) ws returns exn ->
    ret_ok cont_rel over_rel returns ret rt rt' -> cont_pa_ok statics rt ->
    way_out cont_rel over_rel [exn] et et' ->
    statics_rel statics statics' ->
    codes_ok is_local G noted deps cds keep codes symbols False statics ->
    statics_flow_ statics ->
    result_flow (call_function callee Indirect vs a ret rt et statics hp rcd) ->
    result_rel (call_function callee Indirect vs a ret rt et statics hp rcd)
      (call_function callee' Indirect vs' a ret rt' et' statics' hp rcd').

Lemma over_rel_inv (n0 : node) (c c' : cont_value) :
  over_rel n0 c c' ->
  ∃ (rem rem' : list value) (ra ret : arity) (rt rt' et et' : cont_value)
    (ws returns : list node) (exn : node),
    c = Over_application rem ra ret rt et
    ∧ c' = Over_application rem' ra ret rt' et'
    ∧ exact = false ∧ indirect_codes ∧ hu n0 ∧ ra <> []
    ∧ length ws = length ra
    ∧ args_rel (from_node n0) ra rem rem'
    ∧ call_reads (from_node n0) ws returns exn
    ∧ ret_ok cont_rel over_rel returns ret rt rt'
    ∧ way_out cont_rel over_rel [exn] et et'.
Proof.
  intros H.
  destruct H as [n0 rem rem' ra ret rt rt' et et' ws returns exn H1 H2 H3 H4
                   H5 H6 H7 H8 H9].
  exists rem, rem', ra, ret, rt, rt', et, et', ws, returns, exn.
  repeat (split; [ assumption || reflexivity | ]). assumption.
Qed.

(* Values given to a way out: what [apply_cont_answered] does with them,
   and, for what is left of an over-application, what [Hover] says. *)
Lemma returned_cases (c : cont_value) :
  (∀ (n0 : node) (c' : cont_value) (v v' : value)
     (statics statics' : static_env) (hp : heap) (rcd rcd' : read_record),
     over_rel n0 c c' -> rel v n0 v' -> pa_ok statics v ->
     cont_pa_ok statics c ->
     statics_rel statics statics' ->
     codes_ok is_local G noted deps cds keep codes symbols False statics ->
     statics_flow_ statics ->
     result_flow (apply_cont_value c [v] statics hp rcd) ->
     result_rel (apply_cont_value c [v] statics hp rcd)
       (apply_cont_value c' [v'] statics' hp rcd')) ->
  returned_ok c.
Proof.
  intros Hover_ok c' ns vs vs' statics statics' hp rcd rcd' Hway [Hlen_vs
    [Hkinds Hat]]
    Hst Hcodes Hsflow Hvs_pa Hc_pa Hflow'.
  destruct Hway as [(k0 & Hres & Hres' & Hunit_k & Hon)
                   | [(mask & c'' & Hret & Hcrel & Hc')
                     | [(n & mask & h & h' & extras & extras' & -> & Hres
                         & Hres' & Hret & Hcrel & Hhd & Hkx & Hx)
                       | (n & n0 & ns0 & -> & [Hhu0 Htr0] & Hover)]]].
  - (* An exit: one value, at a node with any usage. *)
    subst c c'. cbn [apply_cont_value result_rel].
    eexists; eexists. split; [ reflexivity | ].
    destruct Hon as [Hlen Hon].
    destruct (unit_conts_lookup _ Hunit_k) as (p & Hp & Hpin).
    assert (Hun : unit_cont_nodes u k0 = [p])
      by (unfold unit_cont_nodes; rewrite Hp; reflexivity).
    rewrite Hun in Hlen, Hon.
    destruct ns as [ | n [ | n2 ns]]; try discriminate Hlen.
    destruct vs as [ | w [ | w2 vs]]; try discriminate Hlen_vs.
    destruct vs' as [ | w' [ | w2' vs']]; try discriminate Hkinds.
    destruct (Hon 0 n p eq_refl eq_refl) as [Hn Hnp].
    pose proof (Hat 0 n w w' eq_refl eq_refl eq_refl (or_introl Hn)) as Hrel.
    pose proof (any_usage_input _ _ _ Hc _ (Hunit _ p Hpin)) as Hau.
    destruct (sim_observed sol (exact = true) True w w'
                (rel_au_sim G sol keep called (exact = true) True is_local Hc
                   Hlocal _ _ _ Hau (Hnp w w' Hrel)))
      as [Ho He].
    split; [ split; [ exact Ho | exact He ] | exact I ].
  - (* A handler that code returns to. *)
    assert (Hargs : ∀ (i : nat) (p : variable) (v v' : value),
              nth_error (cont_params c) i = Some (Node_variable p) ->
              nth_error vs i = Some v -> nth_error vs' i = Some v' ->
              is_var_used sol p (kind_of_value v) = true ->
              rel v (Node_variable p) v').
    { intros i p v v' Hp Hv Hv' Hu.
      assert (Hi : i < length ns).
      { rewrite <- Hlen_vs. apply nth_error_Some. congruence. }
      destruct (nth_error ns i) as [n | ] eqn:Hn;
        [ | apply nth_error_None in Hn; lia ].
      destruct (Hret i n (Node_variable p) Hn Hp) as [Hhu Htr].
      apply Htr. apply (Hat i n v v' Hn Hv Hv').
      destruct (is_var_used_cases p _ Hu) as [H | H];
        [ left; exact (Hhu H) | right; exact H ]. }
    pose proof (handler_entered mask c c'' vs vs' statics statics' hp rcd rcd'
      Hcrel
                  Hst Hkinds Hargs) as Hent.
    revert Hflow' Hent. unfold result_flow, result_rel.
    destruct (apply_cont_value c vs statics hp rcd) as [s' | | ];
      intros Hflow' Hent; [ | contradiction | exact I ].
    destruct (Hent Hflow') as (Hkc & t' & Hstep & Hrel').
    destruct Hc' as [[-> Hall] | (kh & vars_w & conts_w & -> & Hkh)].
    + right. left. exists t'. split; [ | exact Hrel' ].
      rewrite (filter_kept_true _ _ Hall) in Hstep. exact Hstep.
    + (* The wrapper, then the handler. *)
      right. right.
      unfold wrapper. cbn [apply_cont_value hm_params_kinds
                           make_handler_metadata].
      rewrite Hkinds, Hkc, kinds_eqb_refl.
      assert (Hlen_w : length (names_from (length (cont_kinds c)) first_fresh)
                       = length vs').
      { rewrite names_from_length, <- Hkc, <- Hkinds, !length_map.
        reflexivity. }
      destruct (bind_params_some _ vs' vars_w Hlen_w) as (wv & Hwv).
      rewrite Hwv. cbv beta iota.
      eexists _, t', 0. split; [ reflexivity | split; [ | exact Hrel' ] ].
      cbn [run]. rewrite step_apply_cont. cbn [ac_args ac_continuation].
      rewrite (eval_simples_filter_kept wv mask _ vs'
                 (eval_bound_params _ vs' vars_w wv (names_from_nodup _ _)
                    Hwv)).
      unfold apply_continuation.
      change (rec_closures [] vars_w conts_w ++ conts_w) with conts_w.
      unfold resolve. rewrite Hkh, Hstep. reflexivity.
  - (* A handler that code raises to, with the extra arguments. *)
    assert (Hl1 : length vs = 1) by (rewrite Hlen_vs; reflexivity).
    destruct vs as [ | w [ | w2 vs]]; try discriminate Hl1.
    destruct vs' as [ | w' [ | w2' vs']]; try discriminate Hkinds.
    subst c c'.
    rewrite apply_with_extra_args in Hflow' |- *.
    rewrite apply_with_extra_args. cbn [app] in Hflow' |- *.
    rewrite <- (filter_kept_hd mask w' extras' Hhd).
    assert (Hkw : map kind_of_value (w' :: extras')
                  = map kind_of_value (w :: extras)).
    { cbn [map] in Hkinds |- *. injection Hkinds as Hk0.
      rewrite Hk0, Hkx. reflexivity. }
    assert (Hargs : ∀ (i : nat) (p : variable) (v v' : value),
              nth_error (cont_params h) i = Some (Node_variable p) ->
              nth_error (w :: extras) i = Some v ->
              nth_error (w' :: extras') i = Some v' ->
              is_var_used sol p (kind_of_value v) = true ->
              rel v (Node_variable p) v').
    { intros [ | i] p v v' Hp Hv Hv' Hu.
      - injection Hv as <-. injection Hv' as <-.
        destruct (Hret 0 n (Node_variable p) eq_refl Hp) as [Hhu Htr].
        apply Htr. apply (Hat 0 n w w' eq_refl eq_refl eq_refl).
        destruct (is_var_used_cases p _ Hu) as [H | H];
          [ left; exact (Hhu H) | right; exact H ].
      - exact (Hx i p v v' Hp Hv Hv' Hu). }
    pose proof (handler_entered mask h h' (w :: extras) (w' :: extras')
                  statics statics' hp rcd rcd' Hcrel Hst Hkw Hargs) as Hent.
    revert Hflow' Hent. unfold result_flow, result_rel.
    destruct (apply_cont_value h (w :: extras) statics hp rcd) as [s' | | ];
      intros Hflow' Hent; [ | contradiction | exact I ].
    right. left. destruct (Hent Hflow') as (_ & t' & Hstep & Hrel').
    exists t'. split; [ exact Hstep | exact Hrel' ].
  - (* What is left of an over-application: a call of the value. *)
    destruct (over_rel_inv _ _ _ Hover)
      as (rem & rem' & ra & ret & rt & rt' & et & et' & ws & returns & exn
          & -> & -> & Hover').
    pose proof (proj1 (proj2 (proj2 Hover'))) as Hhu_n0.
    destruct vs as [ | v [ | v2 vs]]; [ exact I | | exact I ].
    destruct vs' as [ | v' [ | v2' vs']]; try discriminate Hkinds.
    assert (Hrel_v : rel v n0 v').
    { apply Htr0. apply (Hat 0 n v v' eq_refl eq_refl eq_refl).
      left. exact (Hhu0 Hhu_n0). }
    exact (Hover_ok n0 _ v v' statics statics' hp rcd rcd' Hover Hrel_v (proj1
      Hvs_pa)
             Hc_pa Hst Hcodes Hsflow Hflow').
Qed.

Lemma map_fst_combine_eq {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map fst (combine l l') = l.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] H;
    try discriminate H; [ reflexivity | ].
  injection H as H. cbn [combine map fst]. rewrite (IH l' H). reflexivity.
Qed.

Lemma nth_error_poisoned (rd : list param_decision)
    (kinds : list flambda_kind) (vs : list value) (i : nat) (v : value) :
  nth_error (poisoned rd kinds vs) i = Some v ->
  ∃ (d : param_decision) (kind : flambda_kind),
    nth_error rd i = Some d ∧ nth_error kinds i = Some kind
    ∧ ((is_kept d = true ∧ nth_error vs i = Some v)
       ∨ (is_kept d = false ∧ v = Value_const (Poison kind))).
Proof.
  unfold poisoned. rewrite nth_error_map.
  destruct (nth_error (combine (combine rd kinds) vs) i)
    as [[[d kind] w] | ] eqn:E; [ | discriminate ].
  intros Hv. injection Hv as Hv.
  destruct (nth_error_combine_inv _ _ _ _ _ E) as [Hdk Hw].
  destruct (nth_error_combine_inv _ _ _ _ _ Hdk) as [Hd Hk].
  exists d, kind. split; [ exact Hd | split; [ exact Hk | ] ].
  destruct (is_kept d); [ left; subst; auto | right; auto ].
Qed.

(* The values of simples of the kinds [K] are of these kinds. *)
Lemma simples_of_kinds_eval (E : finite_map flambda_kind) (vars vars' : env) :
  env_ok E vars vars' ->
  ∀ (args : list simple) (K : list flambda_kind) (vs : list value),
    simples_of_kinds E args K = true -> eval_simples vars args = Some vs ->
    map kind_of_value vs = K.
Proof.
  intros Henv. induction args as [ | a args IH];
    intros [ | k K] vs Hk Hvs; cbn [simples_of_kinds] in Hk;
    try discriminate Hk.
  - cbn [eval_simples] in Hvs. injection Hvs as <-. reflexivity.
  - cbn [eval_simples] in Hvs.
    destruct (simple_kind E a) as [k' | ] eqn:Hsk; [ | discriminate Hk ].
    apply andb_true_iff in Hk as [Hkk Hk]. apply kind_eqb_eq in Hkk.
    subst k'.
    destruct (eval_simple vars a) as [v | ] eqn:Hv; [ | discriminate Hvs ].
    destruct (eval_simples vars args) as [vs0 | ] eqn:Hvs0;
      [ | discriminate Hvs ].
    injection Hvs as <-. cbn [map]. rewrite (IH K vs0 Hk eq_refl). f_equal.
    destruct a as [x | sym | c]; cbn [simple_kind eval_simple] in Hsk, Hv.
    + destruct (Henv x v Hv) as [Hx _]. rewrite Hx in Hsk.
      injection Hsk as Hsk. exact Hsk.
    + injection Hv as <-. injection Hsk as <-. reflexivity.
    + injection Hv as <-. injection Hsk as <-. reflexivity.
Qed.

(* Values given to a way out that has no use at one of the nodes it is
   given a value at: the first run does not exit. A way out of the unit
   has a use at all, and what is left of an over-application at the
   first, and takes one value. *)
Lemma way_out_no_exit (nsw : list node) (c c'' : cont_value)
    (vs : list value) (statics : static_env) (hp : heap) (rcd : read_record) :
  way_out cont_rel over_rel nsw c c'' -> length vs = length nsw ->
  (∃ (i : nat) (n : node), nth_error nsw i = Some n ∧ ~ hu n) ->
  ∀ (k0 : continuation) (vs0 : list value) (hp0 : heap)
      (rcd0 : read_record),
    apply_cont_value c vs statics hp rcd <> Exit k0 vs0 hp0 rcd0.
Proof.
  intros Hway Hlen (i0 & n0 & Hn0 & Hnu0) k0 vs0 hp0 rcd0 Hex.
  assert (Hcl : ∀ (h : continuation_handler) (hv : env) (hc : cont_env)
                  (r : list (continuation * continuation_handler))
                  (ws : list value),
            apply_cont_value (Closure h hv hc r) ws statics hp rcd
            <> Exit k0 vs0 hp0 rcd0).
  { intros [params md handler] hv hc r ws.
    cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value ws) (hm_params_kinds md));
      [ | discriminate ].
    destruct (bind_params params ws hv); discriminate. }
  destruct Hway
    as [(k1 & -> & _ & _ & [Hlen_o Hon])
       | [(mask & c2 & _ & Hcrel & _)
         | [(n & mask & h & h' & ex & ex' & -> & -> & _ & _ & Hcrel & _)
           | (n & n1 & ns1 & -> & [Hhu1 _] & Hover)]]].
  - destruct (nth_error (unit_cont_nodes u k1) i0) as [p | ] eqn:Hp.
    + exact (Hnu0 (proj1 (Hon i0 n0 p Hn0 Hp))).
    + apply nth_error_None in Hp.
      assert (i0 < length nsw) by (apply nth_error_Some; congruence).
      lia.
  - assert (Hc2 : ∃ (h : continuation_handler) (hv : env) (hc : cont_env)
                    (r : list (continuation * continuation_handler)),
               c = Closure h hv hc r)
      by (destruct Hcrel; eexists _, _, _, _; reflexivity).
    destruct Hc2 as (h & hv & hc & r & ->).
    exact (Hcl h hv hc r vs Hex).
  - rewrite apply_with_extra_args in Hex.
    assert (Hc2 : ∃ (h0 : continuation_handler) (hv : env) (hc : cont_env)
                    (r : list (continuation * continuation_handler)),
               h = Closure h0 hv hc r)
      by (destruct Hcrel; eexists _, _, _, _; reflexivity).
    destruct Hc2 as (h0 & hv & hc & r & ->).
    exact (Hcl h0 hv hc r _ Hex).
  - destruct (over_rel_inv _ _ _ Hover)
      as (rem & rem' & ra & ret & rt & rt' & et & et' & ws & returns
          & exn & -> & _ & Hover').
    pose proof (proj1 (proj2 (proj2 Hover'))) as Hhu_n1.
    destruct vs as [ | v [ | v2 vs]]; try discriminate Hex.
    destruct i0 as [ | i0 ].
    + cbn [nth_error] in Hn0. injection Hn0 as <-.
      exact (Hnu0 (Hhu1 Hhu_n1)).
    + destruct ns1 as [ | n2 ns1];
        [ destruct i0; discriminate Hn0 | cbn [length] in Hlen; lia ].
Qed.

(* The same of the return continuation of code that changes, where it
   does not keep one result, or has no use at one of its nodes. *)
Lemma ret_rel_no_exit (m : list bool) (K : list flambda_kind)
    (ns : list node) (c c' : cont_value) :
  ret_rel cont_rel over_rel m K ns c c' ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rcd : read_record),
    length vs = length ns ->
    (∃ i : nat, nth_error m i = Some false)
    ∨ (∃ (i : nat) (n : node), nth_error ns i = Some n ∧ ~ hu n) ->
    ∀ (k0 : continuation) (vs0 : list value) (hp0 : heap)
      (rcd0 : read_record),
      apply_cont_value c vs statics hp rcd <> Exit k0 vs0 hp0 rcd0.
Proof.
  intros Hrr.
  induction Hrr as [m K ns c c' Hrr
                   | m0 K ns ns0 c k rd vars conts Hin IH Lk L2 Lns Hex Htr0
                       Hkeep Hdel];
    intros vs statics hp rcd Hlen Hw k0 vs0 hp0 rcd0.
  2: { apply (IH vs statics hp rcd); [ rewrite Hlen; symmetry; exact L2 | ].
       destruct Hex as (i & [Hm | Hd]); [ left; exists i; exact Hm | right ].
       destruct (nth_error ns0 i) as [n | ] eqn:Hn.
       - exists i, n. split; [ exact Hn | ].
         exact (proj1 (Hdel i Hd) n Hn).
       - exfalso. apply nth_error_None in Hn.
         assert (i < length rd) by (apply nth_error_Some; congruence).
         lia. }
  assert (Hcl : ∀ (h : continuation_handler) (hv : env) (hc : cont_env)
                  (r : list (continuation * continuation_handler)),
            apply_cont_value (Closure h hv hc r) vs statics hp rcd
            <> Exit k0 vs0 hp0 rcd0).
  { intros [params md handler] hv hc r.
    cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds md));
      [ | discriminate ].
    destruct (bind_params params vs hv); discriminate. }
  destruct Hrr
    as [[Hall Hway] | [(hm & Hret & Hcrel & ->)
       | [(k & hm & rd & kinds & wvars & wconts & c'' & -> & Hk & Hret
           & Hcrel & _)
         | (k & rd & wvars & wconts & nsw & -> & Hway & -> & Lk & Lnsw & Lns
            & (i0 & Hi0) & Htrw & Hkeep & Hdel)]]].
  - destruct Hw as [(i & Hi) | Hw].
    + exfalso. rewrite forallb_forall in Hall.
      pose proof (Hall false (nth_error_In _ _ Hi)) as H. discriminate H.
    + exact (way_out_no_exit ns c c' vs statics hp rcd Hway Hlen Hw k0 vs0 hp0
      rcd0).
  - assert (Hc2 : ∃ (h : continuation_handler) (hv : env) (hc : cont_env)
                    (r : list (continuation * continuation_handler)),
               c = Closure h hv hc r)
      by (destruct Hcrel; eexists _, _, _, _; reflexivity).
    destruct Hc2 as (h & hv & hc & r & ->). exact (Hcl h hv hc r).
  - assert (Hc2 : ∃ (h : continuation_handler) (hv : env) (hc : cont_env)
                    (r : list (continuation * continuation_handler)),
               c = Closure h hv hc r)
      by (destruct Hcrel; eexists _, _, _, _; reflexivity).
    destruct Hc2 as (h & hv & hc & r & ->). exact (Hcl h hv hc r).
  - apply (way_out_no_exit nsw c _ vs statics hp rcd Hway);
      [ rewrite Hlen; symmetry; exact Lnsw | ].
    destruct (nth_error nsw i0) as [n0 | ] eqn:Hn0.
    + exists i0, n0. split; [ exact Hn0 | ].
      exact (proj1 (Hdel i0 Hi0) n0 Hn0).
    + exfalso. apply nth_error_None in Hn0.
      assert (i0 < length rd) by (apply nth_error_Some; congruence).
      lia.
Qed.

(* Values given to the return continuation of code that changes its
   calling convention: the second run gives the results kept to what it
   has, a way out, the rebuilt handler, or the wrapper and then the
   handler. *)
Lemma ret_answered (m : list bool) (K : list flambda_kind) (ns : list node)
    (c c' : cont_value)
    (vs vs' : list value) (statics statics' : static_env) (hp : heap)
    (rcd rcd' : read_record) :
  ret_rel cont_rel over_rel m K ns c c' ->
  returned_ok c ->
  map kind_of_value vs = K ->
  at_nodes ns vs vs' ->
  statics_rel statics statics' ->
  codes_ok is_local G noted deps cds keep codes symbols False statics ->
  statics_flow_ statics ->
  every (pa_ok statics) vs -> cont_pa_ok statics c ->
  result_flow (apply_cont_value c vs statics hp rcd) ->
  result_rel (apply_cont_value c vs statics hp rcd)
    (apply_cont_value c' (filter_kept m vs') statics' hp rcd').
Proof.
  intros Hrr. revert vs'.
  induction Hrr as [m K ns c c' Hrr
                   | m0 K ns ns0 c k rd wvars wconts Hin IH Lk L2 Lns Hex
                       Htr0 Hkeep Hdel];
    intros vs' Hret_c HK Hat Hst Hcodes Hsflow Hvs_pa Hc_pa Hflow'.
  2: {
    (* The wrapper of a call in tail position, then the return
       continuation of the code it is in, which keeps other results. *)
    destruct Hat as [Hlen_vs [Hkinds Hat]].
    assert (Hkv' : map kind_of_value vs' = K) by (rewrite Hkinds; exact HK).
    assert (Lv' : length vs' = length rd).
    { rewrite Lk, <- Hkv', length_map. reflexivity. }
    set (vs_p := poisoned rd K vs').
    assert (Hkp : map kind_of_value vs_p = map kind_of_value vs).
    { unfold vs_p. rewrite (map_kind_poisoned rd K vs' Lk Hkv').
      symmetry. exact HK. }
    assert (Hat' : at_nodes ns0 vs vs_p).
    { split; [ rewrite Hlen_vs; symmetry; exact L2 | ].
      split; [ exact Hkp | ].
      intros i n v v' Hn Hv Hv' Hu.
      destruct (nth_error_poisoned _ _ _ _ _ Hv')
        as (d & kind & Hd & Hkind & [[Hkept Hv''] | [Hkept ->]]).
      - destruct (nth_error ns i) as [n1 | ] eqn:Hn1.
        2: { exfalso. apply nth_error_None in Hn1.
             assert (i < length ns0) by (apply nth_error_Some; congruence).
             lia. }
        destruct (Htr0 i n1 n Hn1 Hn) as [Hhu Htr].
        apply Htr. apply (Hat i n1 v v' Hn1 Hv Hv'').
        destruct Hu as [Hu | Hu]; [ left; exact (Hhu Hu) | right; exact Hu ].
      - exfalso. destruct d as [kd | ]; [ discriminate Hkept | ].
        destruct (Hdel i Hd) as [Hnu Hnk].
        assert (Hvk : kind_of_value v = kind).
        { rewrite <- HK in Hkind. rewrite nth_error_map, Hv in Hkind.
          injection Hkind as Hkind. exact Hkind. }
        destruct (Hnk kind Hkind) as [Hr1 Hr2].
        destruct Hu as [Hu | [Hu | Hu]];
          [ exact (Hnu n Hn Hu) | congruence | congruence ]. }
    pose proof (IH vs_p Hret_c HK Hat' Hst Hcodes Hsflow Hvs_pa Hc_pa Hflow')
      as Hres.
    assert (Hnx : ∀ (k0 : continuation) (vs0 : list value) (hp0 : heap)
      (rcd0 : read_record),
              apply_cont_value c vs statics hp rcd <> Exit k0 vs0 hp0 rcd0).
    { apply (ret_rel_no_exit _ _ _ _ _ Hin vs statics hp rcd).
      - rewrite Hlen_vs. symmetry. exact L2.
      - destruct Hex as (i & [Hm | Hd]); [ left; exists i; exact Hm | right ].
        destruct (nth_error ns0 i) as [n | ] eqn:Hn.
        + exists i, n. split; [ exact Hn | ].
          exact (proj1 (Hdel i Hd) n Hn).
        + exfalso. apply nth_error_None in Hn.
          assert (i < length rd) by (apply nth_error_Some; congruence).
          lia. }
    unfold cc_wrapper. rewrite (map_fst_combine_eq rd K Lk).
    cbn [apply_cont_value hm_params_kinds make_handler_metadata].
    assert (Hkk : kept_kinds rd
                  = map kind_of_value (filter_kept (map is_kept rd) vs')).
    { rewrite <- filter_kept_map, Hkv'.
      exact (kept_kinds_filter rd K Lk Hkeep). }
    rewrite <- Hkk, kinds_eqb_refl.
    assert (Hlen_w : length (names_from (length (kept_kinds rd)) first_fresh)
                     = length (filter_kept (map is_kept rd) vs')).
    { rewrite names_from_length, Hkk, length_map. reflexivity. }
    destruct (bind_params_some _ _ wvars Hlen_w) as (wv & Hwv).
    rewrite Hwv. cbv beta iota.
    apply result_rel_after; [ exact Hnx | ].
    rewrite step_apply_cont. cbn [ac_args ac_continuation].
    rewrite (wrapper_args_eval rd K first_fresh (pad_mask (length rd) m0)
               vs' wvars wv (eq_sym Lk) Lv' (length_pad_mask _ _) Hwv).
    unfold apply_continuation.
    change (rec_closures [] wvars wconts ++ wconts) with wconts.
    fold vs_p.
    assert (Lp : length vs_p = length rd).
    { unfold vs_p, poisoned. rewrite length_map, !length_combine. lia. }
    rewrite <- Lp, filter_kept_pad. exact Hres. }
  destruct Hrr
    as [[Hall Hway] | [(hm & Hret & Hcrel & ->)
       | [(k & hm & rd & kinds & wvars & wconts & c'' & -> & Hk & Hret
           & Hcrel & -> & Hkinds_c & Lk & Hkeep & Hdel)
         | (k & rd & wvars & wconts & nsw & -> & Hway & -> & Lk & Lnsw & _
            & (i0 & Hi0) & Htrw & Hkeep & Hdel)]]].
  - rewrite (filter_kept_true _ _ Hall).
    exact (Hret_c c' ns vs vs' statics statics' hp rcd rcd' Hway
             Hat Hst Hcodes Hsflow Hvs_pa Hc_pa Hflow').
  - destruct Hat as [Hlen_vs [Hkinds Hat]].
    assert (Hargs : ∀ (i : nat) (p : variable) (v v' : value),
              nth_error (cont_params c) i = Some (Node_variable p) ->
              nth_error vs i = Some v -> nth_error vs' i = Some v' ->
              is_var_used sol p (kind_of_value v) = true ->
              rel v (Node_variable p) v').
    { intros i p v v' Hp Hv Hv' Hu.
      assert (Hi : i < length ns).
      { rewrite <- Hlen_vs. apply nth_error_Some. congruence. }
      destruct (nth_error ns i) as [n | ] eqn:Hn;
        [ | apply nth_error_None in Hn; lia ].
      destruct (Hret i n (Node_variable p) Hn Hp) as [Hhu Htr].
      apply Htr. apply (Hat i n v v' Hn Hv Hv').
      destruct (is_var_used_cases p _ Hu) as [H | H];
        [ left; exact (Hhu H) | right; exact H ]. }
    pose proof (handler_entered hm c c' vs vs' statics statics' hp rcd rcd'
      Hcrel
                  Hst Hkinds Hargs) as Hent.
    assert (Lv' : length vs' = length ns).
    { rewrite <- Hlen_vs, <- (length_map kind_of_value vs'), Hkinds,
        length_map.
      reflexivity. }
    rewrite <- Lv', filter_kept_pad.
    revert Hflow' Hent. unfold result_flow, result_rel.
    destruct (apply_cont_value c vs statics hp rcd) as [s' | | ];
      intros Hflow' Hent; [ | contradiction | exact I ].
    destruct (Hent Hflow') as (_ & t' & Hstep & Hrel').
    right. left. exists t'. split; [ exact Hstep | exact Hrel' ].
  - destruct Hat as [Hlen_vs [Hkinds Hat]].
    subst kinds. pose proof HK as Hkc.
    assert (Hkv' : map kind_of_value vs' = K)
      by (rewrite Hkinds; exact Hkc).
    assert (Lv' : length vs' = length rd).
    { rewrite Lk, <- Hkv', length_map. reflexivity. }
    set (vs_p := poisoned rd K vs').
    assert (Hkp : map kind_of_value vs_p = map kind_of_value vs).
    { unfold vs_p. rewrite (map_kind_poisoned rd K vs' Lk Hkv').
      symmetry. exact Hkc. }
    assert (Hargs : ∀ (i : nat) (p : variable) (v v' : value),
              nth_error (cont_params c) i = Some (Node_variable p) ->
              nth_error vs i = Some v -> nth_error vs_p i = Some v' ->
              is_var_used sol p (kind_of_value v) = true ->
              rel v (Node_variable p) v').
    { intros i p v v' Hp Hv Hv' Hu.
      destruct (nth_error_poisoned _ _ _ _ _ Hv')
        as (d & kind & Hd & Hkind & [[Hkept Hv''] | [Hkept ->]]).
      - assert (Hi : i < length ns).
        { rewrite <- Hlen_vs. apply nth_error_Some. congruence. }
        destruct (nth_error ns i) as [n | ] eqn:Hn;
          [ | apply nth_error_None in Hn; lia ].
        destruct (Hret i n (Node_variable p) Hn Hp) as [Hhu Htr].
        apply Htr. apply (Hat i n v v' Hn Hv Hv'').
        destruct (is_var_used_cases p _ Hu) as [H | H];
          [ left; exact (Hhu H) | right; exact H ].
      - exfalso. destruct d as [kd | ]; [ discriminate Hkept | ].
        assert (Hvk : kind_of_value v = kind).
        { rewrite <- Hkc in Hkind. rewrite nth_error_map, Hv in Hkind.
          injection Hkind as Hkind. exact Hkind. }
        rewrite Hvk, (Hdel i p kind Hd Hp Hkind) in Hu. discriminate Hu. }
    pose proof (handler_entered hm c c'' vs vs_p statics statics' hp rcd rcd'
      Hcrel
                  Hst Hkp Hargs) as Hent.
    revert Hflow' Hent. unfold result_flow, result_rel.
    destruct (apply_cont_value c vs statics hp rcd) as [s' | | ];
      intros Hflow' Hent; [ | contradiction | exact I ].
    destruct (Hent Hflow') as (_ & t' & Hstep & Hrel').
    (* The wrapper, then the handler. *)
    right. right.
    unfold cc_wrapper. rewrite (map_fst_combine_eq rd K Lk).
    cbn [apply_cont_value hm_params_kinds make_handler_metadata].
    assert (Hkk : kept_kinds rd
                  = map kind_of_value (filter_kept (map is_kept rd) vs')).
    { rewrite <- filter_kept_map, Hkv'.
      exact (kept_kinds_filter rd K Lk Hkeep). }
    rewrite <- Hkk, kinds_eqb_refl.
    assert (Hlen_w : length (names_from (length (kept_kinds rd)) first_fresh)
                     = length (filter_kept (map is_kept rd) vs')).
    { rewrite names_from_length, Hkk, length_map. reflexivity. }
    destruct (bind_params_some _ _ wvars Hlen_w) as (wv & Hwv).
    rewrite Hwv. cbv beta iota.
    eexists _, t', 0. split; [ reflexivity | split; [ | exact Hrel' ] ].
    cbn [run]. rewrite step_apply_cont. cbn [ac_args ac_continuation].
    rewrite (wrapper_args_eval rd K first_fresh
               (pad_mask (length rd) hm) vs' wvars wv (eq_sym Lk) Lv'
               (length_pad_mask _ _) Hwv).
    unfold apply_continuation.
    change (rec_closures [] wvars wconts ++ wconts) with wconts.
    unfold resolve. rewrite Hk. fold vs_p.
    assert (Lp : length vs_p = length rd).
    { unfold vs_p, poisoned. rewrite length_map, !length_combine. lia. }
    rewrite <- Lp, filter_kept_pad, Hstep. reflexivity.
  - (* The wrapper of a call in tail position, then the way out, which
       does not exit: a way out of the unit, or what is left of an
       over-application given one value, has a use at every node that it
       is given a value at. *)
    destruct Hat as [Hlen_vs [Hkinds Hat]].
    assert (Hkv' : map kind_of_value vs' = K) by (rewrite Hkinds; exact HK).
    assert (Lv' : length vs' = length rd).
    { rewrite Lk, <- Hkv', length_map. reflexivity. }
    assert (Lns : length ns = length rd).
    { rewrite <- Hlen_vs, <- (length_map kind_of_value vs), <- Hkinds,
        length_map.
      exact Lv'. }
    assert (Hnw0 : ∃ n0 : node, nth_error nsw i0 = Some n0 ∧ ~ hu n0).
    { destruct (nth_error nsw i0) as [n0 | ] eqn:Hn0.
      - exists n0. split; [ reflexivity | exact (proj1 (Hdel i0 Hi0) n0 Hn0) ].
      - exfalso. apply nth_error_None in Hn0.
        assert (i0 < length rd) by (apply nth_error_Some; congruence).
        lia. }
    destruct Hnw0 as (n0 & Hn0 & Hnu0).
    set (vs_p := poisoned rd K vs').
    assert (Hkp : map kind_of_value vs_p = map kind_of_value vs).
    { unfold vs_p. rewrite (map_kind_poisoned rd K vs' Lk Hkv').
      symmetry. exact HK. }
    assert (Hat' : at_nodes nsw vs vs_p).
    { split; [ rewrite Hlen_vs; symmetry; exact Lnsw | ].
      split; [ exact Hkp | ].
      intros i n v v' Hn Hv Hv' Hu.
      destruct (nth_error_poisoned _ _ _ _ _ Hv')
        as (d & kind & Hd & Hkind & [[Hkept Hv''] | [Hkept ->]]).
      - destruct (nth_error ns i) as [n1 | ] eqn:Hn1.
        2: { exfalso. apply nth_error_None in Hn1.
             assert (i < length nsw) by (apply nth_error_Some; congruence).
             lia. }
        destruct (Htrw i n1 n Hn1 Hn) as [Hhu Htr].
        apply Htr. apply (Hat i n1 v v' Hn1 Hv Hv'').
        destruct Hu as [Hu | Hu]; [ left; exact (Hhu Hu) | right; exact Hu ].
      - exfalso. destruct d as [kd | ]; [ discriminate Hkept | ].
        destruct (Hdel i Hd) as [Hnu Hnk].
        assert (Hvk : kind_of_value v = kind).
        { rewrite <- HK in Hkind. rewrite nth_error_map, Hv in Hkind.
          injection Hkind as Hkind. exact Hkind. }
        destruct (Hnk kind Hkind) as [Hr1 Hr2].
        destruct Hu as [Hu | [Hu | Hu]];
          [ exact (Hnu n Hn Hu) | congruence | congruence ]. }
    pose proof (Hret_c (resolve k wconts) nsw vs vs_p
                  statics statics' hp rcd rcd' Hway Hat' Hst Hcodes Hsflow
                    Hvs_pa
                  Hc_pa Hflow')
      as Hres.
    (* The first run does not exit. *)
    assert (Hnx : ∀ (k0 : continuation) (vs0 : list value) (hp0 : heap)
      (rcd0 : read_record),
              apply_cont_value c vs statics hp rcd <> Exit k0 vs0 hp0 rcd0).
    { intros k0 vs0 hp0 rcd0 Hex.
      assert (Hcl : ∀ (h : continuation_handler) (hv : env) (hc : cont_env)
                      (r : list (continuation * continuation_handler))
                      (ws : list value),
                apply_cont_value (Closure h hv hc r) ws statics hp rcd
                <> Exit k0 vs0 hp0 rcd0).
      { intros [params md handler] hv hc r ws.
        cbn [apply_cont_value].
        destruct (kinds_eqb (map kind_of_value ws) (hm_params_kinds md));
          [ | discriminate ].
        destruct (bind_params params ws hv); discriminate. }
      destruct Hway
        as [(k1 & -> & _ & _ & [Hlen_o Hon])
           | [(mask & c2 & _ & Hcrel & _)
             | [(n & mask & h & h' & ex & ex' & -> & -> & _ & _ & Hcrel & _)
               | (n & n1 & ns1 & -> & [Hhu1 _] & Hover)]]].
      - destruct (nth_error (unit_cont_nodes u k1) i0) as [p | ] eqn:Hp.
        + exact (Hnu0 (proj1 (Hon i0 n0 p Hn0 Hp))).
        + apply nth_error_None in Hp.
          assert (i0 < length nsw) by (apply nth_error_Some; congruence).
          lia.
      - assert (Hc2 : ∃ (h : continuation_handler) (hv : env)
                        (hc : cont_env)
                        (r : list (continuation * continuation_handler)),
                   c = Closure h hv hc r)
          by (destruct Hcrel; eexists _, _, _, _; reflexivity).
        destruct Hc2 as (h & hv & hc & r & ->).
        exact (Hcl h hv hc r vs Hex).
      - rewrite apply_with_extra_args in Hex.
        assert (Hc2 : ∃ (h0 : continuation_handler) (hv : env)
                        (hc : cont_env)
                        (r : list (continuation * continuation_handler)),
                   h = Closure h0 hv hc r)
          by (destruct Hcrel; eexists _, _, _, _; reflexivity).
        destruct Hc2 as (h0 & hv & hc & r & ->).
        exact (Hcl h0 hv hc r _ Hex).
      - destruct (over_rel_inv _ _ _ Hover)
          as (rem & rem' & ra & ret & rt & rt' & et & et' & ws & returns
              & exn & -> & _ & Hover').
        pose proof (proj1 (proj2 (proj2 Hover'))) as Hhu_n1.
        destruct vs as [ | v [ | v2 vs]]; try discriminate Hex.
        assert (Lrd1 : length rd = 1)
          by (rewrite <- Lns, <- Hlen_vs; reflexivity).
        destruct i0 as [ | i0 ].
        + cbn [nth_error] in Hn0. injection Hn0 as <-.
          exact (Hnu0 (Hhu1 Hhu_n1)).
        + assert (S i0 < length rd) by (apply nth_error_Some; congruence).
          lia. }
    (* The wrapper, then the way out. *)
    unfold cc_wrapper. rewrite (map_fst_combine_eq rd K Lk).
    cbn [apply_cont_value hm_params_kinds make_handler_metadata].
    assert (Hkk : kept_kinds rd
                  = map kind_of_value (filter_kept (map is_kept rd) vs')).
    { rewrite <- filter_kept_map, Hkv'.
      exact (kept_kinds_filter rd K Lk Hkeep). }
    rewrite <- Hkk, kinds_eqb_refl.
    assert (Hlen_w : length (names_from (length (kept_kinds rd)) first_fresh)
                     = length (filter_kept (map is_kept rd) vs')).
    { rewrite names_from_length, Hkk, length_map. reflexivity. }
    destruct (bind_params_some _ _ wvars Hlen_w) as (wv & Hwv).
    rewrite Hwv. cbv beta iota.
    apply result_rel_after; [ exact Hnx | ].
    rewrite step_apply_cont. cbn [ac_args ac_continuation].
    rewrite (wrapper_args_eval rd K first_fresh (pad_mask (length rd) [])
               vs' wvars wv (eq_sym Lk) Lv' (length_pad_mask _ _) Hwv).
    unfold apply_continuation.
    change (rec_closures [] wvars wconts ++ wconts) with wconts.
    fold vs_p.
    assert (Lp : length vs_p = length rd).
    { unfold vs_p, poisoned. rewrite length_map, !length_combine. lia. }
    rewrite <- Lp, filter_kept_pad, filter_kept_nil. exact Hres.
Qed.

(* Values given to a way out, or to the return continuation of code that
   changes that keeps every result. *)
Lemma returned_either (c : cont_value) :
  returned_ok c ->
  ∀ (c' : cont_value) (ns : list node) (vs vs' : list value)
    (statics statics' : static_env) (hp : heap) (rcd rcd' : read_record),
    way_out cont_rel over_rel ns c c'
    ∨ ret_rel cont_rel over_rel (repeat true (length ns))
        (map kind_of_value vs) ns c c' ->
    at_nodes ns vs vs' ->
    statics_rel statics statics' ->
    codes_ok is_local G noted deps cds keep codes symbols False statics ->
    statics_flow_ statics ->
    every (pa_ok statics) vs -> cont_pa_ok statics c ->
    result_flow (apply_cont_value c vs statics hp rcd) ->
    result_rel (apply_cont_value c vs statics hp rcd)
      (apply_cont_value c' vs' statics' hp rcd').
Proof.
  intros Hrc c' ns vs vs' statics statics' hp rcd rcd' [Hw | Hr] Hat Hst Hcodes
    Hsflow
    Hvs_pa Hc_pa Hflow'.
  - exact (Hrc c' ns vs vs' statics statics' hp rcd rcd' Hw Hat Hst Hcodes
    Hsflow
             Hvs_pa Hc_pa Hflow').
  - pose proof (ret_answered _ _ _ _ _ vs vs' statics statics' hp rcd rcd' Hr
    Hrc
                  eq_refl Hat Hst Hcodes Hsflow Hvs_pa Hc_pa Hflow') as H.
    rewrite filter_kept_true in H; [ exact H | apply forallb_repeat_true ].
Qed.


(* The call of a closure of the rebuilt code, in the second run, has the
   shape of the call in the first. *)
Lemma shape_rebuilt (statics statics' : static_env) (closure closure' : value)
    (id : code_id) (slots slots' : list (value_slot * value)) (c : code)
    (all all' : list value) (A ret : arity) (shape : call_shape) :
  unchanged id c ->
  closure_of statics closure = Some (id, slots) ->
  lookup id (static_code statics) = Some c ->
  closure_of statics' closure' = Some (id, slots') ->
  lookup id (static_code statics') = Some (rebuild_c id c) ->
  code_is_tupled c = false ->
  map kind_of_value all' = map kind_of_value all ->
  shape_of_closure_call statics closure Indirect all A ret = Some shape ->
  shape_of_closure_call statics' closure' Indirect all' A ret
  = Some
      (match shape with
       | Full_call _ _ _ _ => Full_call closure' (rebuild_c id c) all' A
       | Over_call _ _ _ _ a1 a2 =>
         Over_call closure' (rebuild_c id c)
           (firstn (cardinal_unarized (code_params_arity c)) all')
           (skipn (cardinal_unarized (code_params_arity c)) all') a1 a2
       | Partial_call _ =>
         Partial_call (Value_partial_application closure' all' A)
       end).
Proof.
  intros Hun Hcl Hcode Hcl' Hcode' Htup Hk Hshape.
  destruct (code_fields_rebuild id c Hun)
    as (_ & _ & _ & _ & Epa & Era & Et).
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (id0 & slots0 & c0 & all0 & A0 & Hcl0 & Hcode0 & _ & Hafc & Hcall).
  rewrite Hcl in Hcl0. injection Hcl0 as <- <-.
  rewrite Hcode in Hcode0. injection Hcode0 as <-.
  rewrite (arguments_for_code_not_tupled _ _ _ _ Htup) in Hafc.
  injection Hafc as <- <-.
  assert (Ht' : code_is_tupled (rebuild_c id c) = false)
    by (rewrite Et; exact Htup).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[Hfull ->] | [(Hover & Hkinds & Hpoo & ->)
                     | (Hpart & Hkinds & Hpoo & _ & ->)]].
  - apply (shape_of_closure_call_full _ _ _ _ _ _ id slots' _ Hcl'
             (or_introl eq_refl) Hcode' Ht').
    rewrite Epa. exact Hfull.
  - assert (Hn' : length A <> length (code_params_arity (rebuild_c id c)))
      by (rewrite Epa; lia).
    rewrite (shape_of_closure_call_not_full _ _ _ _ _ _ id slots' _ Hcl'
               (or_introl eq_refl) Hcode' Ht' Hn').
    rewrite Epa, Era, Hk, Hkinds, kinds_eqb_refl, Hpoo. cbn [andb negb].
    destruct (Nat.ltb_spec (length (code_params_arity c)) (length A))
      as [_ | Hge]; [ reflexivity | lia ].
  - assert (Hn' : length A <> length (code_params_arity (rebuild_c id c)))
      by (rewrite Epa; lia).
    rewrite (shape_of_closure_call_not_full _ _ _ _ _ _ id slots' _ Hcl'
               (or_introl eq_refl) Hcode' Ht' Hn').
    rewrite Epa, Era, Hk, Hkinds, kinds_eqb_refl, Hpoo. cbn [andb negb].
    destruct (Nat.ltb_spec (length (code_params_arity c)) (length A))
      as [Hlt | _]; [ lia | reflexivity ].
Qed.

(* A full call of the rebuilt code. *)
Lemma call_code_rebuilt (closure' : value) (id : code_id) (c : code)
    (vs' : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics' : static_env) (hp : heap) (rcd' : read_record)
    (callee_vars' : env) :
  unchanged id c ->
  binds_statics (code_body (rebuild_c id c)) = false ->
  kinds_eqb (map kind_of_value vs') (unarize args_arity) = true ->
  kinds_per_parameter_eqb (unarize_per_parameter args_arity)
    (unarize_per_parameter (code_params_arity c)) = true ->
  arity_eqb return_arity (code_result_arity c) = true ->
  bind_params (code_params c) vs' [(code_my_closure c, closure')]
  = Some callee_vars' ->
  call_code closure' (rebuild_c id c) vs' args_arity return_arity return_to
    exn_to statics' hp rcd'
  = Step (State (code_body (rebuild_c id c)) callee_vars'
            [(code_return_continuation c, return_to);
             (code_exn_continuation c, exn_to)]
            statics' hp rcd').
Proof.
  intros Hun Hbinds Hk1 Hk2 Hk3 Hbind.
  destruct (code_fields_rebuild id c Hun)
    as (Ep & Em & Er & Ee & Epa & Era & Et).
  unfold call_code. rewrite Hbinds, Hk1, Epa, Hk2, Era, Hk3. cbn [negb andb].
  rewrite Ep, Em, Hbind, Er, Ee. reflexivity.
Qed.

Lemma forallb_unarize_skipn (f : flambda_kind -> bool) (p : nat) (A : arity) :
  forallb f (unarize A) = true -> forallb f (unarize (skipn p A)) = true.
Proof.
  rewrite <- (firstn_skipn p A) at 1. unfold unarize.
  rewrite flat_map_app, forallb_app. intros H.
  apply andb_true_iff in H as [_ H]. exact H.
Qed.

Lemma forallb_unarize_app (f : flambda_kind -> bool) (a1 a2 : arity) :
  forallb f (unarize a1) = true -> forallb f (unarize a2) = true ->
  forallb f (unarize (a1 ++ a2)) = true.
Proof.
  unfold unarize. rewrite flat_map_app, forallb_app. intros -> ->.
  reflexivity.
Qed.

Lemma map_firstn_comm {A B : Type} (f : A -> B) (n : nat) (l : list A) :
  map f (firstn n l) = firstn n (map f l).
Proof.
  revert l. induction n as [ | n IH]; intros [ | x l]; simpl;
    [ reflexivity | reflexivity | reflexivity | f_equal; apply IH ].
Qed.

Lemma result_flow_of_ok (r : step_result) :
  result_ok is_local G sol noted deps cds keep codes symbols u False no_loc
    r ->
  result_flow r.
Proof.
  destruct r as [s' | | ]; [ exact (fun H => H) | intros _; exact I
                           | intros _; exact I ].
Qed.

(* The call of a closure of code that is not tupled, with the arguments
   that a partial application of it keeps ([applied], at [M]) and those
   of the call ([vs], at [N]): a full call, an over-application, or a
   partial application, answered by the call of the rebuilt code in the
   second run. *)
Lemma closure_called (rt rt' et et' : cont_value) (closure closure' : value)
    (M N : node) (applied applied' vs vs' : list value) (aa a ret : arity)
    (ws returns : list node) (exn : node) (statics statics' : static_env)
    (hp : heap) (rcd rcd' : read_record) (shape : call_shape) :
  returned_ok rt ->
  exact = false -> indirect_codes -> a <> [] ->
  length ws = length a ->
  rel closure M closure' -> hu M -> pa_ok statics closure ->
  (∀ (g : nat) (W : node),
     wchain (from_node M) (length aa + g) W -> wchain (from_node N) g W) ->
  args_rel (from_node M) aa applied applied' ->
  args_rel (from_node N) a vs vs' ->
  length applied = length (unarize aa) ∨ map kind_of_value vs = unarize a ->
  (∀ (id : code_id) (slots : list (value_slot * value)) (c : code),
     closure_of statics closure = Some (id, slots) ->
     lookup id (static_code statics) = Some c ->
     length aa < length (code_params_arity c)
     ∧ code_is_tupled c = false) ->
  call_reads (from_node N) ws returns exn ->
  ret_ok cont_rel over_rel returns ret rt rt' -> cont_pa_ok statics rt ->
  way_out cont_rel over_rel [exn] et et' ->
  every (pa_ok statics) (applied ++ vs) ->
  statics_rel statics statics' ->
  codes_ok is_local G noted deps cds keep codes symbols False statics ->
  statics_flow_ statics ->
  shape_of_closure_call statics closure Indirect (applied ++ vs) (aa ++ a) ret
  = Some shape ->
  result_flow
    (run_call (fun partial => apply_cont_value rt [partial] statics hp rcd)
       (Some shape) ret rt et statics hp rcd) ->
  result_rel
    (run_call (fun partial => apply_cont_value rt [partial] statics hp rcd)
       (Some shape) ret rt et statics hp rcd)
    (run_call (fun partial => apply_cont_value rt' [partial] statics' hp rcd')
       (shape_of_closure_call statics' closure' Indirect (applied' ++ vs')
          (aa ++ a) ret)
       ret rt' et' statics' hp rcd').
Proof.
  intros Hret_ok Hexf Hunt Hne Hlen_ws Hrel_c HhuM Hcl_pa Hshift Haa Ha
    Hlen_or Hjk Hreads Hret Hrt_pa Hexn Hall_pa Hst Hcodes Hsflow Hshape
    Hflow'.
  assert (Hnl_code : is_local Field_code_id_of_call_witness = false)
    by (apply (not_local is_local Hlocal); intros; discriminate).
  assert (Hnl_exn : is_local (Field_return_of_call Exn) = false)
    by (apply (not_local is_local Hlocal); intros; discriminate).
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (id & slots & c & all0 & A0 & Hclosure & Hcode & _ & Hafc & Hcall).
  destruct (code_facts is_local G noted deps cds keep codes symbols False Hnd
              statics id c Hcodes Hcode)
    as (Hin_codes & cd & Hcd & Hunk & _ & Lp & Lr).
  destruct (Hjk id slots c Hclosure Hcode) as [Hlt_aa Htupled].
  assert (Hne_c : code_params_arity c <> [])
    by (intros E; rewrite E in Hlt_aa; simpl in Hlt_aa; lia).
  rewrite (arguments_for_code_not_tupled _ _ _ _ Htupled) in Hafc.
  injection Hafc as <- <-.
  destruct (rel_closure_of statics statics' closure closure' M id slots Hst
              Hrel_c HhuM Hclosure)
    as (slots' & Hclosure').
  destruct (rel_closure_alloc statics closure closure' M id slots Hsflow
              Hrel_c Hclosure)
    as (b & Hb & Hrel_b & Hcalled_b).
  destruct (Hcds id cd Hcd) as (_ & _ & _ & _ & Hasid).
  pose proof (called_alloc is_local G sol cds Hc Hcds b id Hcalled_b) as Hbb.
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & _ & Hcw).
  destruct (Hunk Htupled) as [Lu Hcch].
  rewrite (group_by_parameter_groups _ _ Hne_c) in Lu, Hcch.
  rewrite length_group_by_sizes, length_arity_sizes in Lu.
  destruct (Hcode_checks id c Hin_codes)
    as (Hscr_c & Hcf_c & Hraises_c & Hclosed_c & Hkinded_c & Hfresh_c).
  pose proof (Hcodes id c Hcode) as [_ Hedges_c].
  assert (Hhs_id : hs (Node_code_id id))
    by exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hasid)).
  set (L := combine
              (group_by_sizes (arity_sizes (code_params_arity c))
                 (cd_params cd))
              (cd_unknown_arity_call_witnesses cd)) in *.
  assert (Hwc : ∀ (g : nat) (first : list node) (W : node),
            nth_error L g = Some (first, W) -> wchain (from_node M) g W).
  { apply (code_wchain id (cd_return cd) (cd_exn cd) _ _ Hcch Hhs_id).
    intros first W Hhd. exists b.
    split; [ split; [ exact Hb | exact Hbb ] | ].
    apply Hcw. exact (hd_error_combine _ _ _ _ Hhd). }
  assert (HLc : length L = length (code_params_arity c)).
  { unfold L. rewrite length_combine, Lu, length_group_by_sizes,
      length_arity_sizes.
    apply Nat.min_id. }
  assert (Hnth_L : ∀ g : nat,
            g < length (code_params_arity c) ->
            ∃ (f : list node) (W : node), nth_error L g = Some (f, W)).
  { intros g Hg. destruct (nth_error L g) as [[f W] | ] eqn:E; [ eauto | ].
    apply nth_error_None in E. lia. }
  pose proof Hreads as Hreads0.
  destruct Hreads as (Hcover & Hlast_ret & Hexn_acc & Hnext_acc
                      & (called_node & Hau_called & Hcalled_acc) & _).
  assert (Hnth_ws : ∀ g : nat,
            g < length a -> ∃ w : node, nth_error ws g = Some w).
  { intros g Hg. destruct (nth_error ws g) as [w | ] eqn:E; [ eauto | ].
    apply nth_error_None in E. lia. }
  (* A witness of the code at or after the groups that the partial
     application keeps stands for one of the call. *)
  assert (HcovL : ∀ (g : nat) (f : list node) (W w : node),
            nth_error L (length aa + g) = Some (f, W) ->
            nth_error ws g = Some w -> may_come_from sol W w).
  { intros g f W w HL Hw. apply (Hcover g W w); [ | exact Hw ].
    apply Hshift. exact (Hwc _ f W HL). }
  (* The code is used where a witness of the call stands for one of it. *)
  assert (Hlive_of : ∀ (g : nat) (f : list node) (W w : node),
            nth_error L (length aa + g) = Some (f, W) ->
            nth_error ws g = Some w -> au (Node_code_id id)).
  { intros g f W w HL Hw.
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hcch HL) as (_ & Hcodeid & _).
    exact (read_any_usage _ _ _ Hc W w called_node _ _ Hnl_code Hcodeid
             (Hcalled_acc g w Hw) (HcovL g f W w HL Hw) Hau_called). }
  (* The code, as it is bound. *)
  destruct c as [params mc rk ek md cbody].
  cbn [code_body code_params code_my_closure code_return_continuation
       code_exn_continuation code_params_arity code_result_arity
       code_is_tupled] in *.
  destruct Hedges_c as (cd' & Hcd' & _ & _ & _ & _ & _ & Hmc & Hps & Hne_k
                        & Hbinds & Hcounted & Hbody).
  rewrite (lookup_nodup_in _ _ _ Hnd Hcd') in Hcd. injection Hcd as Hcd.
  subst cd'.
  assert (Hcode'_of : au (Node_code_id id) ->
            lookup id (static_code statics')
            = Some (rebuild_c id (Code params mc rk ek md cbody))).
  { intros Hlive'. pose proof Hst as (_ & _ & _ & Hsc).
    destruct (Hsc id) as [H | [_ H]].
    - rewrite H, Hcode. reflexivity.
    - exfalso. exact (H (au_hu _ _ _ Hc _ Hlive')). }
  (* The arguments together, at the witnesses of the code. *)
  assert (Hlen_app : map kind_of_value (applied ++ vs) = unarize (aa ++ a) ->
            length applied = length (unarize aa)).
  { intros Hk. destruct Hlen_or as [Hl | Hk2]; [ exact Hl | ].
    apply (f_equal (@length _)) in Hk. apply (f_equal (@length _)) in Hk2.
    rewrite length_map, length_app, length_unarize_app in Hk.
    rewrite length_map in Hk2. lia. }
  assert (HA_of : map kind_of_value (applied ++ vs) = unarize (aa ++ a) ->
            args_rel (from_node M) (aa ++ a) (applied ++ vs)
              (applied' ++ vs')).
  { intros Hk.
    exact (args_rel_app _ _ aa a applied applied' vs vs' Haa Ha Hshift
             (Hlen_app Hk)). }
  (* The parameters of the code get the arguments at them. *)
  assert (Hparams :
    ∀ (all all' : list value) (A : arity),
      args_rel (from_node M) A all all' ->
      map kind_of_value all = unarize A ->
      (∀ i g kk : nat,
         group_position (arity_sizes (cm_params_arity md)) i = Some (g, kk)
         -> group_position (arity_sizes A) i = Some (g, kk)) ->
      length params = length (cd_params cd) ->
      ∀ (i : nat) (p : variable) (v v' : value),
        nth_error params i = Some p -> nth_error all i = Some v ->
        nth_error all' i = Some v' ->
        is_var_used sol p (kind_of_value v) = true ->
        nth_error (repeat true (length params)) i = Some true
        ∧ rel v (Node_variable p) v').
  { intros all all' A HA Hkall Hpos_eq Hlenp i p v v' Hp Hv Hv' Hu.
    assert (Hi : i < length params) by (apply nth_error_Some; congruence).
    split; [ exact (nth_error_repeat_lt true _ _ Hi) | ].
    destruct (argument_kind_ok (kind_of_value v)) eqn:Hkv.
    2: { (* The poison of [Region] or [Rec_info], the same in the two
            runs. *)
         exact (rel_region_argument all all' v v' i _ Hkv (proj1 HA) Hv Hv'). }
    assert (Hhu_p : hu (Node_variable p)).
    { destruct (kind_of_value v); simpl in Hkv, Hu;
        try discriminate Hkv; exact Hu. }
    assert (Hi' : i < length (cd_params cd)) by (rewrite <- Hlenp; exact Hi).
    destruct (nth_error (cd_params cd) i) as [q | ] eqn:Hq;
      [ | apply nth_error_None in Hq; lia ].
    pose proof (alias_input _ _ _ Hc _ _
                  (Hps p q (nth_error_combine _ _ _ _ _ Hp Hq))) as Hal_pq.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_p Hal_pq) as Hhu_q.
    assert (Hiq : i < list_sum (arity_sizes (cm_params_arity md))).
    { rewrite sum_arity_sizes, <- Lp. exact Hi'. }
    destruct (group_position_some _ _ Hiq) as (g & kk & Hpos).
    destruct (group_by_sizes_nth _ _ _ _ _ _ Hpos Hq) as (grp & Hgrp & Hq').
    destruct (group_position_lt _ _ _ _ Hpos) as [Hg _].
    rewrite length_arity_sizes in Hg.
    destruct (Hnth_L g Hg) as (f & W & HLg).
    destruct (nth_error_combine_inv _ _ _ _ _ HLg) as [Hf _].
    rewrite Hgrp in Hf. injection Hf as <-.
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hcch HLg) as (Hpar & _).
    apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _ _
             _ Hal_pq).
    exact (args_rel_nth _ _ _ _ HA i v v' g kk W q Hv Hv' (Hpos_eq _ _ _ Hpos)
             (Hwc g grp W HLg) (Hpar kk q Hq') Hhu_q). }
  (* The code is used: the first witness of the call reads its code id. *)
  destruct (Hnth_L (length aa + 0) ltac:(lia)) as (f0 & W0 & HL0).
  destruct (Hnth_ws 0 ltac:(destruct a; [ contradiction | simpl; lia ]))
    as (w0 & Hw0).
  pose proof (Hlive_of 0 f0 W0 w0 HL0 Hw0) as Hlive'.
  pose proof (Hcode'_of Hlive') as Hcode'.
  assert (Hkind_c : kind_of_value closure = Value).
  { destruct closure; try discriminate Hclosure; reflexivity. }
  assert (Henv0_of : au (Node_code_id id) ->
            env_ok [(mc, Value)] [(mc, closure)] [(mc, closure')]).
  { intros Hlive0.
    assert (Hrel_mc : rel closure (Node_variable mc) closure').
    { apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _
               _ _ (alias_input _ _ _ Hc _ _ Hmc)).
      apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _
               _ _ (alias_propagate _ _ _ Hc _ _ _ Hlive0 Hprop)).
      exact Hrel_b. }
    intros x v Hx. simpl in Hx |- *.
    destruct (Pos.eqb_spec x mc) as [-> | _]; [ | discriminate Hx ].
    injection Hx as <-. rewrite Hkind_c. split; [ reflexivity | ].
    intros _. exists closure'. split; [ reflexivity | exact Hrel_mc ]. }
  (* The code keeps its calling convention: the call uses its witness of
     unknown arity. *)
  assert (Hun : unchanged id (Code params mc rk ek md cbody)).
  { pose proof (Hshift 0 W0 (Hwc _ f0 W0 HL0)) as HW0.
    pose proof (call_reads_hu _ _ _ _ Hreads0 0 W0 w0 HW0 Hw0) as Hhu_W0.
    destruct HW0 as (y & _ & Hcons & _).
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hcch HL0) as (_ & Hcodeid0 & _).
    exact (unchanged_of_unknown id _ y W0 Hcons Hhu_W0 Hcodeid0). }
  pose proof Lr as Lr0.
  destruct (code_fields_rebuild id (Code params mc rk ek md cbody) Hun)
    as (_ & _ & _ & _ & _ & Era & _).
  cbn [code_result_arity] in Era.
  pose proof (shape_of_code_call_cases _ _ _ _ _ _ Hcall) as Hcases.
  cbn [code_params_arity code_result_arity code_is_tupled] in Hcases.
  destruct Hcases
    as [[Hfull ->] | [(Hover & Hkinds & Hpoo & ->)
                     | (Hpart & Hkinds & Hpoo & _ & ->)]].
  - (* A full call. *)
    cbn [run_call] in Hflow' |- *.
    revert Hflow'.
    match goal with
    | |- result_flow ?r -> _ =>
      destruct r as [s' | k' vs'' hp'' rcd'' | ] eqn:Hcc
    end;
      intros Hflow';
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & Hk1 & Hk2 & Hk3 & Hbind & ->).
    cbn [code_body code_params code_my_closure code_return_continuation
         code_exn_continuation code_params_arity code_result_arity]
      in Hk2, Hk3, Hbind.
    pose proof (HA_of Hk1) as HA.
    pose proof (args_rel_length _ _ _ _ HA) as Hlen'.
    (* The last witness of the code, and that of the call. *)
    assert (Hla : S (pred (length a)) = length a)
      by (destruct a; [ contradiction | reflexivity ]).
    rewrite length_app in Hfull.
    destruct (Hnth_L (length aa + pred (length a)) ltac:(lia))
      as (fl & Wl & HLl).
    destruct (Hnth_ws (pred (length a)) ltac:(lia)) as (wl & Hwl).
    pose proof (HcovL _ fl Wl wl HLl Hwl) as HWl.
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hcch HLl)
      as (_ & _ & Hlast_c & _).
    destruct (Hlast_c ltac:(rewrite HLc; lia)) as [Hexnc Hretc].
    assert (Hwl' : nth_error ws (pred (length ws)) = Some wl)
      by (rewrite Hlen_ws; exact Hwl).
    assert (Hlenp : length params = length (cd_params cd)).
    { rewrite Lp, <- (unarize_per_parameter_eq _ _ Hk2), <- Hk1, length_map.
      symmetry. exact (bind_params_length _ _ _ _ Hbind). }
    assert (Hpos_eq : ∀ i g kk : nat,
              group_position (arity_sizes (cm_params_arity md)) i
              = Some (g, kk) ->
              group_position (arity_sizes (aa ++ a)) i = Some (g, kk)).
    { intros i g kk H. unfold arity_sizes. rewrite Hk2. exact H. }
    destruct (bind_params_rel params (applied ++ vs) (applied' ++ vs')
                (repeat true (length params)) [(mc, Value)] [(mc, closure)]
                [(mc, closure')] callee_vars (Henv0_of Hlive') Hlen'
                (repeat_length _ _)
                (Hparams _ _ _ HA Hk1 Hpos_eq Hlenp) Hbind)
      as (callee_vars' & Hbind' & Hok').
    rewrite !filter_kept_all in Hbind'.
    rewrite (shape_rebuilt statics statics' closure closure' id slots slots'
               (Code params mc rk ek md cbody) (applied ++ vs)
               (applied' ++ vs') (aa ++ a) ret _ Hun Hclosure Hcode Hclosure'
               Hcode' Htupled (proj1 HA) Hshape).
    cbn [run_call].
    right. left. eexists. split.
    + eapply (call_code_rebuilt closure' id (Code params mc rk ek md cbody)
               _ _ _ _ _ _ _ _ _ Hun);
        cbn [code_params code_my_closure code_params_arity code_result_arity
             code_body].
      * apply rebuilt_code_no_statics; assumption.
      * rewrite (proj1 HA), Hk1. apply kinds_eqb_refl.
      * apply kinds_per_parameter_eqb_eq. exact Hk2.
      * apply arity_eqb_eq. exact Hk3.
      * exact Hbind'.
    + assert (Hway_e : way_out cont_rel over_rel [cd_exn cd] et et').
      { apply (way_out_transport _ _ [exn]); [ exact Hexn | reflexivity | ].
        intros i n n' Hn Hn'. destruct i as [ | [ | i]]; try discriminate Hn.
        injection Hn as <-. injection Hn' as <-.
        exact (read_transports _ _ Wl wl _ Hnl_exn Hexnc
                 (Hexn_acc _ wl Hwl) HWl). }
      assert (Lret : length (cd_return cd) = length (unarize ret))
        by (rewrite Lr, <- Hk3; reflexivity).
      assert (Htr_r : ∀ (i : nat) (n n' : node),
                nth_error returns i = Some n ->
                nth_error (cd_return cd) i = Some n' -> transports n' n).
      { intros i n n' Hn Hn'.
        exact (read_transports _ _ Wl wl _ (result_not_local i)
                 (Hretc i n' Hn') (Hlast_ret i wl n Hwl' Hn) HWl). }
      destruct (proj1 Hret) as [Hret_w | Hret_r].
      * exact (code_entered id params mc rk ek md cbody cd closure
                 (applied ++ vs) callee_vars callee_vars' rt rt' et et'
                 statics statics' hp rcd rcd' Hun Hflow' Hst Hscr_c Hcf_c
                   Hraises_c
                 Hclosed_c Hkinded_c Hfresh_c Hne_k Hcounted Hbody Lr
                 (eq_trans Hk1 (unarize_per_parameter_eq _ _ Hk2)) Hbind Hok'
                 Hlive' (Hret_w _ Lret Htr_r) Hway_e).
      * apply (code_entered_ret id params mc rk ek md cbody cd closure
                 (applied ++ vs) callee_vars callee_vars' rt rt' et et'
                 statics statics' hp rcd rcd' Hun Hflow' Hst Hscr_c Hcf_c
                   Hraises_c
                 Hclosed_c Hkinded_c Hfresh_c
                 (proj1 (Hcode_rets id params mc rk ek md cbody Hexf
                           Hin_codes) Hne_k)
                 (proj2 (Hcode_rets id params mc rk ek md cbody Hexf
                           Hin_codes))
                 Hne_k Hcounted Hbody Lr
                 (eq_trans Hk1 (unarize_per_parameter_eq _ _ Hk2)) Hbind Hok'
                 Hlive'); [ | exact Hway_e ].
        rewrite <- Hk3. exact (Hret_r _ Lret Htr_r).
  - (* An over-application. *)
    cbn [run_call] in Hflow' |- *.
    revert Hflow'.
    match goal with
    | |- result_flow ?r -> _ =>
      destruct r as [s' | k' vs'' hp'' rcd'' | ] eqn:Hcc
    end;
      intros Hflow';
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & Hk1 & Hk2 & _ & Hbind & ->).
    cbn [code_body code_params code_my_closure code_return_continuation
         code_exn_continuation code_params_arity code_result_arity]
      in Hk2, Hbind.
    pose proof (HA_of Hkinds) as HA.
    pose proof (args_rel_length _ _ _ _ HA) as Hlen'.
    rewrite length_app in Hover.
    set (p := length (cm_params_arity md)) in *.
    set (size := cardinal_unarized (cm_params_arity md)) in *.
    set (j := p - length aa).
    (* The last witness of the code, and the witness of the call of that
       group. *)
    destruct (Hnth_L (pred p) ltac:(lia)) as (fl & Wl & HLl).
    assert (HLl' : nth_error L (length aa + pred j) = Some (fl, Wl))
      by (replace (length aa + pred j) with (pred p) by (unfold j; lia);
          exact HLl).
    destruct (Hnth_ws (pred j) ltac:(unfold j; lia)) as (wl & Hwl).
    pose proof (HcovL _ fl Wl wl HLl' Hwl) as HWl.
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hcch HLl)
      as (_ & _ & Hlast_c & _).
    destruct (Hlast_c ltac:(rewrite HLc; lia)) as [Hexnc Hretc].
    (* The code returns one value. *)
    unfold partial_or_over_application in Hpoo.
    apply andb_true_iff in Hpoo as [_ Hpoo].
    rewrite length_app in Hpoo. fold p in Hpoo.
    destruct (Nat.ltb_spec p (length aa + length a)) as [_ | Hge]; [ | lia ].
    apply is_one_param_of_kind_value_eq in Hpoo.
    assert (Ecr : ∃ R0 : node, cd_return cd = [R0]).
    { rewrite Hpoo in Lr. simpl in Lr.
      destruct (cd_return cd) as [ | R0 [ | R1 rr]]; try discriminate Lr.
      exists R0. reflexivity. }
    destruct Ecr as (R0 & Ecr).
    assert (HR0 : nth_error (cd_return cd) 0 = Some R0)
      by (rewrite Ecr; reflexivity).
    assert (Hhu_R0 : hu R0).
    { destruct (Hnth_ws j ltac:(unfold j; lia)) as (wj & Hwj).
      replace j with (S (pred j)) in Hwj by (unfold j; lia).
      destruct (Hnext_acc (pred j) wl wj Hwl Hwj) as (vc & Hvc & Hwjv).
      pose proof (has_usage_accessor _ _ _ Hc _ _ _
                    (au_hu _ _ _ Hc _ Hau_called) (Hcalled_acc _ wj Hwj))
        as Hhu_wj.
      pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_wj Hwjv) as Hhu_vc.
      exact (read_hu _ _ _ _ _ (result_not_local 0) (Hretc 0 R0 HR0) Hvc
               HWl Hhu_vc). }
    (* The arguments left, after the groups of the code. *)
    assert (Hshift_R0 : ∀ (g : nat) (W : node),
              wchain (from_node R0) g W -> wchain (from_node M) (p + g) W).
    { intros g W HW.
      pose proof (wchain_from_return _ _ Wl R0 (Hwc _ fl Wl HLl)
                    (Hretc 0 R0 HR0) g W HW) as H.
      replace (p + g) with (S (pred p) + g) by lia. exact H. }
    assert (Hsizes : arity_sizes (cm_params_arity md)
                     = firstn p (arity_sizes (aa ++ a))).
    { rewrite <- arity_sizes_firstn. unfold arity_sizes. rewrite Hk2.
      reflexivity. }
    assert (Hsize : size = list_sum (firstn p (arity_sizes (aa ++ a)))).
    { unfold size, cardinal_unarized. rewrite <- sum_arity_sizes, Hsizes.
      reflexivity. }
    assert (Hp_le : p <= length (aa ++ a)) by (rewrite length_app; lia).
    pose proof (args_rel_skipn _ _ (aa ++ a) _ _ p HA Hshift_R0 Hp_le)
      as Hrem.
    rewrite <- Hsize in Hrem.
    assert (Hover_rel :
      over_rel R0
        (Over_application (skipn size (applied ++ vs))
           (partially_apply (aa ++ a) p) ret rt et)
        (Over_application (skipn size (applied' ++ vs'))
           (partially_apply (aa ++ a) p) ret rt' et')).
    { apply (Over_rel R0 _ _ _ ret rt rt' et et' (skipn j ws) returns exn
               Hexf Hunt Hhu_R0).
      - unfold partially_apply. intros E. apply (f_equal (@length _)) in E.
        rewrite length_skipn, length_app in E. simpl in E. lia.
      - unfold partially_apply. rewrite !length_skipn, Hlen_ws, length_app.
        unfold j. lia.
      - exact Hrem.
      - split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
        + intros g W w HW Hw. rewrite nth_error_skipn_add in Hw.
          apply (Hcover (j + g) W w); [ | exact Hw ].
          apply Hshift.
          replace (length aa + (j + g)) with (p + g) by (unfold j; lia).
          exact (Hshift_R0 g W HW).
        + intros i w r Hw Hr.
          rewrite nth_error_skipn_last in Hw
            by (rewrite Hlen_ws; unfold j; lia).
          exact (Hlast_ret i w r Hw Hr).
        + intros g w Hw. rewrite nth_error_skipn_add in Hw.
          exact (Hexn_acc _ w Hw).
        + intros g w w' Hw Hw'. rewrite !nth_error_skipn_add in Hw, Hw'.
          rewrite Nat.add_succ_r in Hw'.
          exact (Hnext_acc (j + g) w w' Hw Hw').
        + exists called_node. split; [ exact Hau_called | ].
          intros g w Hw. rewrite nth_error_skipn_add in Hw.
          exact (Hcalled_acc _ w Hw).
        + intros W w HW Hw. rewrite nth_error_skipn_add in Hw.
          apply (call_reads_hu _ _ _ _ Hreads0 (j + 0) W w); [ | exact Hw ].
          apply Hshift.
          replace (length aa + (j + 0)) with (p + 0) by (unfold j; lia).
          exact (Hshift_R0 0 W HW).
      - exact Hret.
      - exact Hexn. }
    (* The parameters get the first arguments. *)
    assert (Hlenp : length params = length (cd_params cd)).
    { rewrite Lp, <- (unarize_per_parameter_eq _ _ Hk2), <- Hk1, length_map.
      symmetry. exact (bind_params_length _ _ _ _ Hbind). }
    assert (Hparams_over :
      ∀ (i : nat) (p0 : variable) (v v' : value),
        nth_error params i = Some p0 ->
        nth_error (firstn size (applied ++ vs)) i = Some v ->
        nth_error (firstn size (applied' ++ vs')) i = Some v' ->
        is_var_used sol p0 (kind_of_value v) = true ->
        nth_error (repeat true (length params)) i = Some true
        ∧ rel v (Node_variable p0) v').
    { intros i p0 v v' Hp0 Hv Hv' Hu.
      assert (Hi : i < size).
      { assert (Hl : i < length (firstn size (applied ++ vs)))
          by (apply nth_error_Some; congruence).
        rewrite length_firstn in Hl. lia. }
      rewrite nth_error_firstn_lt in Hv, Hv' by exact Hi.
      refine (Hparams _ _ _ HA Hkinds _ Hlenp i p0 v v' Hp0 Hv Hv' Hu).
      intros i' g kk H. rewrite Hsizes in H.
      rewrite <- (firstn_skipn p (arity_sizes (aa ++ a))).
      rewrite group_position_app_l; [ exact H | ].
      exact (proj2 (group_position_lt _ _ _ _ H)). }
    assert (Hlen_f : length (firstn size (applied' ++ vs'))
                     = length (firstn size (applied ++ vs)))
      by (rewrite !length_firstn, Hlen'; reflexivity).
    destruct (bind_params_rel params _ _ (repeat true (length params))
                [(mc, Value)] [(mc, closure)] [(mc, closure')] callee_vars
                (Henv0_of Hlive') Hlen_f (repeat_length _ _) Hparams_over
                Hbind)
      as (callee_vars' & Hbind' & Hok').
    rewrite !filter_kept_all in Hbind'.
    rewrite (shape_rebuilt statics statics' closure closure' id slots slots'
               (Code params mc rk ek md cbody) (applied ++ vs)
               (applied' ++ vs') (aa ++ a) ret _ Hun Hclosure Hcode Hclosure'
               Hcode' Htupled (proj1 HA) Hshape).
    cbn [run_call code_params_arity].
    fold p size.
    right. left. eexists. split.
    + eapply (call_code_rebuilt closure' id (Code params mc rk ek md cbody)
               _ _ _ _ _ _ _ _ _ Hun);
        cbn [code_params code_my_closure code_params_arity code_result_arity
             code_body].
      * apply rebuilt_code_no_statics; assumption.
      * rewrite !map_firstn_comm, (proj1 HA), <- map_firstn_comm, Hk1.
        apply kinds_eqb_refl.
      * apply kinds_per_parameter_eqb_eq. exact Hk2.
      * rewrite Era. apply arity_eqb_eq. reflexivity.
      * exact Hbind'.
    + apply (code_entered id params mc rk ek md cbody cd closure
               (firstn size (applied ++ vs)) callee_vars callee_vars' _ _ et
               et' statics statics' hp rcd rcd' Hun Hflow' Hst Hscr_c Hcf_c
                 Hraises_c
               Hclosed_c Hkinded_c Hfresh_c Hne_k Hcounted Hbody Lr0
               (eq_trans Hk1 (unarize_per_parameter_eq _ _ Hk2)) Hbind Hok'
               Hlive').
      * rewrite Ecr. right. right. right. exists R0, R0, [].
        split; [ reflexivity | ].
        split; [ split; [ intros H; exact H | intros v0 v0' H; exact H ] | ].
        exact Hover_rel.
      * apply (way_out_transport _ _ [exn]); [ exact Hexn | reflexivity | ].
        intros i n n' Hn Hn'. destruct i as [ | [ | i]]; try discriminate Hn.
        injection Hn as <-. injection Hn' as <-.
        exact (read_transports _ _ Wl wl _ Hnl_exn Hexnc
                 (Hexn_acc _ wl Hwl) HWl).
  - (* A partial application. *)
    cbn [run_call] in Hflow' |- *.
    pose proof (HA_of Hkinds) as HA.
    unfold partial_or_over_application in Hpoo.
    apply andb_true_iff in Hpoo as [_ Hpoo].
    destruct (Nat.ltb_spec (length (cm_params_arity md)) (length (aa ++ a)))
      as [Hlt | _]; [ lia | ].
    apply andb_true_iff in Hpoo as [_ Hpoo].
    apply is_one_param_of_kind_value_eq in Hpoo. subst ret.
    destruct (proj2 Hret eq_refl) as [Hrne | Hundef].
    2: { rewrite Hundef. exact I. }
    destruct returns as [ | r0 rest]; [ contradiction | ].
    rewrite (shape_rebuilt statics statics' closure closure' id slots slots'
               (Code params mc rk ek md cbody) (applied ++ vs)
               (applied' ++ vs') (aa ++ a) _ _ Hun Hclosure Hcode Hclosure'
               Hcode' Htupled (proj1 HA) Hshape).
    cbn [run_call].
    apply (returned_either rt Hret_ok _ [r0]).
    + assert (Htr0 : ∀ (i : nat) (n n' : node),
                nth_error (r0 :: rest) i = Some n ->
                nth_error [r0] i = Some n' -> transports n' n).
      { intros i n n' Hn Hn'.
        destruct i as [ | i]; [ | destruct i; discriminate Hn' ].
        injection Hn as <-. injection Hn' as <-.
        split; [ intros H; exact H | intros v0 v0' H; exact H ]. }
      destruct (proj1 Hret) as [Hret_w | Hret_r];
        [ left; exact (Hret_w [r0] eq_refl Htr0)
        | right; exact (Hret_r [r0] eq_refl Htr0) ].
    + split; [ reflexivity | split; [ reflexivity | ] ].
      intros [ | i] n v v' Hn Hv Hv' _; [ | destruct i; discriminate Hn ].
      injection Hn as <-. injection Hv as <-. injection Hv' as <-.
      cbn [rel]. split; [ exact I | ].
      exists M, closure', (applied' ++ vs').
      split; [ reflexivity | ].
      split; [ intros E; rewrite Hexf in E; discriminate E | ].
      split; [ exact Hrel_c | split; [ exact HhuM | ] ].
      split; [ exact (rel_closure_observed statics _ _ M id slots Hrel_c
                        Hclosure) | ].
      split.
      { intros E. apply (f_equal (@length _)) in E. rewrite length_app in E.
        destruct a; [ contradiction | simpl in E; lia ]. }
      split; [ | exact HA ].
      intros g W R y Hg HW HR HyR Hyy.
      assert (Hg' : g = length aa + pred (length a)).
      { rewrite length_app in Hg. destruct a; [ contradiction | ].
        simpl in Hg |- *. lia. }
      subst g.
      destruct (Hnth_ws (pred (length a))
                  ltac:(destruct a; [ contradiction | simpl; lia ]))
        as (wl & Hwl).
      pose proof (Hcover _ W wl (Hshift _ W HW) Hwl) as HWwl.
      assert (Hwl' : nth_error ws (pred (length ws)) = Some wl)
        by (rewrite Hlen_ws; exact Hwl).
      exact (may_come_from_read _ _ _ Hc y W wl r0 R _ (result_not_local 0)
               HR (Hlast_ret 0 wl r0 Hwl' eq_refl) HWwl HyR).
    + exact Hst.
    + exact Hcodes.
    + exact Hsflow.
    + split; [ | exact I ].
      split; [ | split; [ exact Hcl_pa | exact Hall_pa ] ].
      exists id, slots, (Code params mc rk ek md cbody).
      split; [ exact Hclosure | split; [ exact Hcode | ] ].
      split; [ exact Hpart | exact Htupled ].
    + exact Hrt_pa.
    + exact Hflow'.
Qed.


(* The call of the rebuilt code, in the second run, has the shape of the
   call of the code in the first, given arguments of the same kinds. *)
Lemma shape_code_rebuilt (closure closure' : value) (id : code_id)
    (c : code)
    (all all' : list value) (A ret : arity) (shape : call_shape) :
  unchanged id c ->
  map kind_of_value all' = map kind_of_value all ->
  shape_of_code_call closure c all A ret = Some shape ->
  shape_of_code_call closure' (rebuild_c id c) all' A ret
  = Some
      (match shape with
       | Full_call _ _ _ _ => Full_call closure' (rebuild_c id c) all' A
       | Over_call _ _ _ _ a1 a2 =>
         Over_call closure' (rebuild_c id c)
           (firstn (cardinal_unarized (code_params_arity c)) all')
           (skipn (cardinal_unarized (code_params_arity c)) all') a1 a2
       | Partial_call _ =>
         Partial_call (Value_partial_application closure' all' A)
       end).
Proof.
  intros Hun Hk Hshape.
  destruct (code_fields_rebuild id c Hun)
    as (_ & _ & _ & _ & Epa & Era & Et).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hshape)
    as [[Hfull ->] | [(Hover & Hkinds & Hpoo & ->)
                     | (Hpart & Hkinds & Hpoo & Hnt & ->)]];
    unfold shape_of_code_call; cbv zeta; rewrite Epa, Era, Et, Hk.
  - rewrite Hfull, Nat.eqb_refl. reflexivity.
  - assert (Hn : Nat.eqb (length A) (length (code_params_arity c)) = false)
      by (apply Nat.eqb_neq; lia).
    rewrite Hn, Hkinds, kinds_eqb_refl, Hpoo. cbn [andb negb].
    destruct (Nat.ltb_spec (length (code_params_arity c)) (length A))
      as [_ | Hge]; [ reflexivity | lia ].
  - assert (Hn : Nat.eqb (length A) (length (code_params_arity c)) = false)
      by (apply Nat.eqb_neq; lia).
    rewrite Hn, Hkinds, kinds_eqb_refl, Hpoo, Hnt. cbn [andb negb].
    destruct (Nat.ltb_spec (length (code_params_arity c)) (length A))
      as [Hlt | _]; [ lia | reflexivity ].
Qed.

(* The call of a closure of tupled code, which takes the fields of the
   tuple, the first argument, followed by the other arguments: a full
   call, or an over-application, answered by the call of the rebuilt
   code. The second run has the tuple: the code is used, as the first
   witness of the call reads its code id, and the two facts that keep
   the tuple alive give the parameter of the witness of the code a use
   ([tupled_kept]), so the mask keeps the tuple, related there. It is a
   block of as many fields, those used related and the others poison,
   and the parameters of the code that have a use get related fields
   ([tuple_fields_rel]). The arguments after the tuple are at the
   witnesses that come after that of the code, from what it returns. *)
Lemma tupled_called (rt rt' et et' : cont_value) (callee callee' : value)
    (N : node) (vs vs' : list value) (a ret : arity) (ws returns : list node)
    (exn : node) (statics statics' : static_env) (hp : heap)
    (rcd rcd' : read_record)
    (shape : call_shape) (id : code_id) (slots : list (value_slot * value))
    (c : code) :
  exact = false -> indirect_codes ->
  length ws = length a ->
  rel callee N callee' -> hu N ->
  args_rel (from_node N) a vs vs' ->
  closure_of statics callee = Some (id, slots) ->
  lookup id (static_code statics) = Some c ->
  code_is_tupled c = true ->
  call_reads (from_node N) ws returns exn ->
  ret_ok cont_rel over_rel returns ret rt rt' ->
  way_out cont_rel over_rel [exn] et et' ->
  statics_rel statics statics' ->
  codes_ok is_local G noted deps cds keep codes symbols False statics ->
  statics_flow_ statics ->
  shape_of_closure_call statics callee Indirect vs a ret = Some shape ->
  result_flow
    (run_call (fun partial => apply_cont_value rt [partial] statics hp rcd)
       (Some shape) ret rt et statics hp rcd) ->
  result_rel
    (run_call (fun partial => apply_cont_value rt [partial] statics hp rcd)
       (Some shape) ret rt et statics hp rcd)
    (run_call (fun partial => apply_cont_value rt' [partial] statics' hp rcd')
       (shape_of_closure_call statics' callee' Indirect vs' a ret)
       ret rt' et' statics' hp rcd').
Proof.
  intros Hexf Hunt Hlen_ws Hrel_c HhuN Ha Hclosure Hcode Htupled Hreads Hret
    Hexn Hst Hcodes Hsflow Hshape Hflow'.
  assert (Hnl_code : is_local Field_code_id_of_call_witness = false)
    by (apply (not_local is_local Hlocal); intros; discriminate).
  assert (Hnl_exn : is_local (Field_return_of_call Exn) = false)
    by (apply (not_local is_local Hlocal); intros; discriminate).
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (id' & slots0 & c' & all0 & A0 & Hclosure0 & Hcode0 & _ & Hafc
        & Hcall).
  rewrite Hclosure in Hclosure0. injection Hclosure0 as <- <-.
  rewrite Hcode in Hcode0. injection Hcode0 as <-.
  destruct (code_facts is_local G noted deps cds keep codes symbols False Hnd
              statics id c Hcodes Hcode)
    as (Hin_codes & cd & Hcd & _ & Htw & Lp & Lr).
  pose proof (Hunt id c Hin_codes) as Hsingle. rewrite Htupled in Hsingle.
  pose proof (singleton_params_length _ Hsingle) as Hsize.
  destruct (Htw Htupled) as (W & U0 & HWs & Hretc & Hexnc & Hcodeid & _).
  destruct (Hkeep_alive id c cd Hin_codes Htupled Hcd)
    as (W' & U & Kn & HWs' & HparW & Hacc & HKacc & HKuse).
  rewrite HWs in HWs'. injection HWs' as <-.
  (* The arguments as the code takes them: the fields of the tuple, then
     the arguments after it. *)
  unfold arguments_for_code in Hafc. rewrite Htupled in Hafc.
  unfold detuple in Hafc.
  destruct vs as [ | tuple rest]; [ discriminate Hafc | ].
  destruct a as [ | first rest_arity]; [ discriminate Hafc | ].
  destruct (kinds_eqb (map kind_of_value (tuple :: rest))
              (unarize (first :: rest_arity))) eqn:Hk1;
    [ | discriminate Hafc ].
  destruct (kinds_eqb (unarize_component first) [Value]) eqn:Hk2;
    [ | discriminate Hafc ].
  cbn [negb andb] in Hafc.
  destruct (tuple_fields statics (cardinal_unarized (code_params_arity c))
              tuple) as [fields | ] eqn:Hfields; [ | discriminate Hafc ].
  injection Hafc as <- <-.
  apply kinds_eqb_eq in Hk1. apply kinds_eqb_eq in Hk2.
  assert (Hlen_rest : length rest = length (unarize rest_arity)).
  { assert (H := f_equal (@length flambda_kind) Hk1). simpl in H.
    rewrite length_map in H. unfold unarize in H. simpl in H.
    rewrite length_app, Hk2 in H. simpl in H. fold (unarize rest_arity) in H.
    lia. }
  assert (Hsizes : arity_sizes (first :: rest_arity)
                   = 1 :: arity_sizes rest_arity).
  { unfold arity_sizes, unarize_per_parameter. simpl. rewrite Hk2.
    reflexivity. }
  destruct vs' as [ | tuple' rest'];
    [ destruct Ha as [Hk _]; discriminate Hk | ].
  assert (Hk_rest : map kind_of_value rest' = map kind_of_value rest).
  { pose proof (proj1 Ha) as H. cbn [map] in H. injection H as _ H.
    exact H. }
  destruct ws as [ | w1 ws']; [ discriminate Hlen_ws | ].
  cbn [length] in Hlen_ws. injection Hlen_ws as Hlen_ws.
  (* The witness of the code, which the first witness of the call stands
     for. *)
  destruct (rel_closure_of statics statics' callee callee' N id slots Hst
              Hrel_c HhuN Hclosure)
    as (slots' & Hclosure').
  destruct (rel_closure_alloc statics callee callee' N id slots Hsflow
              Hrel_c Hclosure)
    as (b & Hb & Hrel_b & Hcalled_b).
  destruct (Hcds id cd Hcd) as (_ & _ & _ & _ & Hasid).
  pose proof (called_alloc is_local G sol cds Hc Hcds b id Hcalled_b) as Hbb.
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & _ & Hcw).
  rewrite HWs in Hcw.
  assert (Hhs_id : hs (Node_code_id id))
    by exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hasid)).
  assert (HW0 : wchain (from_node N) 0 W).
  { exists b. split; [ split; [ exact Hb | exact Hbb ] | ].
    split; [ exact (Hcw W eq_refl) | ].
    exact (alloc_self is_local G sol Hc _ _ _ Hcodeid Hhs_id). }
  pose proof Hreads as Hreads0.
  destruct Hreads as (Hcover & Hlast_ret & Hexn_acc & Hnext_acc
                      & (called_node & Hau_called & Hcalled_acc) & _).
  pose proof (Hcover 0 W w1 HW0 eq_refl) as HWw1.
  (* The code keeps its calling convention: the call uses its witness of
     unknown arity. *)
  assert (Hun : unchanged id c).
  { pose proof (call_reads_hu _ _ _ _ Hreads0 0 W w1 HW0 eq_refl) as Hhu_W.
    pose proof HW0 as (y & _ & Hcons & _).
    exact (unchanged_of_unknown id c y W Hcons Hhu_W Hcodeid). }
  pose proof (read_any_usage _ _ _ Hc W w1 called_node _ _ Hnl_code Hcodeid
                (Hcalled_acc 0 w1 eq_refl) HWw1 Hau_called) as Hlive'.
  (* The tuple, kept alive. *)
  assert (Hhu_U : hu U).
  { pose proof (any_usage_use _ _ _ Hc _ _ (au_hu _ _ _ Hc _ Hlive') HKuse)
      as HauK.
    exact (has_usage_accessor _ _ _ Hc _ _ _ (au_hu _ _ _ Hc _ HauK) HKacc).
  }
  assert (Hrel_t : rel tuple U tuple').
  { refine (args_rel_nth _ _ _ _ Ha 0 tuple tuple' 0 0 W U eq_refl eq_refl
              _ HW0 HparW Hhu_U).
    rewrite Hsizes. reflexivity. }
  destruct (tuple_fields_rel statics statics' _ tuple tuple' U fields Hst
              Hrel_t Hhu_U Hfields)
    as (fields' & Hfields' & Hkv & Hkf & Hrelf).
  destruct (tuple_fields_block _ _ _ _ Hfields) as [Hlen_f _].
  unfold cardinal_unarized in Hlen_f. rewrite Hsize in Hlen_f.
  assert (Hlen_f' : length fields' = length fields).
  { rewrite <- (length_map kind_of_value fields'), Hkf, length_map.
    reflexivity. }
  (* The call of the second run. *)
  assert (Hcode' : lookup id (static_code statics') = Some (rebuild_c id c)).
  { pose proof Hst as (_ & _ & _ & Hsc).
    destruct (Hsc id) as [H | [_ H]].
    - rewrite H, Hcode. reflexivity.
    - exfalso. exact (H (au_hu _ _ _ Hc _ Hlive')). }
  destruct (code_fields_rebuild id c Hun) as (_ & _ & _ & _ & Epa & _ & Et).
  rewrite (FlambdaTupledProofs.shape_of_closure_call_tupled statics' callee'
             Indirect tuple' rest' first rest_arity ret id slots'
             (rebuild_c id c) fields' Hclosure' (or_introl eq_refl) Hcode'
             ltac:(rewrite Et; exact Htupled)
             ltac:(rewrite (proj1 Ha); exact Hk1) Hk2
             ltac:(rewrite Epa; exact Hfields')).
  rewrite Epa.
  assert (Hkall : map kind_of_value (fields' ++ rest')
                  = map kind_of_value (fields ++ rest))
    by (rewrite !map_app, Hkf, Hk_rest; reflexivity).
  rewrite (shape_code_rebuilt callee callee' id c _ _ _ ret shape Hun Hkall
             Hcall).
  (* The code, as it is bound. *)
  pose proof (Hcodes id c Hcode) as [_ Hedges_c].
  destruct (Hcode_checks id c Hin_codes)
    as (Hscr_c & Hcf_c & Hraises_c & Hclosed_c & Hkinded_c & Hfresh_c).
  pose proof (shape_of_code_call_cases _ _ _ _ _ _ Hcall) as Hcases.
  destruct c as [params mc rk ek md cbody].
  cbn [code_body code_params code_my_closure code_return_continuation
       code_exn_continuation code_params_arity code_result_arity
       code_is_tupled] in *.
  destruct Hedges_c as (cd' & Hcd' & _ & _ & _ & _ & _ & Hmc & Hps & Hne_k
                        & Hbinds & Hcounted & Hbody).
  rewrite (lookup_nodup_in _ _ _ Hnd Hcd') in Hcd. injection Hcd as Hcd.
  subst cd'.
  assert (Hkind_c : kind_of_value callee = Value).
  { destruct callee; try discriminate Hclosure; reflexivity. }
  assert (Henv0 : env_ok [(mc, Value)] [(mc, callee)] [(mc, callee')]).
  { assert (Hrel_mc : rel callee (Node_variable mc) callee').
    { apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _
               _ _ (alias_input _ _ _ Hc _ _ Hmc)).
      apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _
               _ _ (alias_propagate _ _ _ Hc _ _ _ Hlive' Hprop)).
      exact Hrel_b. }
    intros x v Hx. simpl in Hx |- *.
    destruct (Pos.eqb_spec x mc) as [-> | _]; [ | discriminate Hx ].
    injection Hx as <-. rewrite Hkind_c. split; [ reflexivity | ].
    intros _. exists callee'. split; [ reflexivity | exact Hrel_mc ]. }
  (* The parameters of the code get the fields of the tuple. *)
  assert (Hparams :
    length params = length (cd_params cd) ->
    ∀ (i : nat) (p : variable) (v v' : value),
      nth_error params i = Some p -> nth_error fields i = Some v ->
      nth_error fields' i = Some v' ->
      is_var_used sol p (kind_of_value v) = true ->
      nth_error (repeat true (length params)) i = Some true
      ∧ rel v (Node_variable p) v').
  { intros Hlenp i p v v' Hp Hv Hv' Hu.
    assert (Hi : i < length params) by (apply nth_error_Some; congruence).
    split; [ exact (nth_error_repeat_lt true _ _ Hi) | ].
    rewrite Forall_forall in Hkv.
    rewrite (Hkv v (nth_error_In _ _ Hv)) in Hu. cbn [is_var_used] in Hu.
    assert (Hi' : i < length (cd_params cd)) by (rewrite <- Hlenp; exact Hi).
    destruct (nth_error (cd_params cd) i) as [q | ] eqn:Hq;
      [ | apply nth_error_None in Hq; lia ].
    pose proof (alias_input _ _ _ Hc _ _
                  (Hps p q (nth_error_combine _ _ _ _ _ Hp Hq))) as Hal_pq.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hu Hal_pq) as Hhu_q.
    apply (rel_alias G sol keep called (exact = true) True is_local Hc _ _ _
             _ Hal_pq).
    exact (Hrelf i q v v' (Hacc i q Hq) Hhu_q Hv Hv'). }
  destruct Hcases
    as [[Hfull ->] | [(Hover & Hkinds & Hpoo & ->)
                     | (_ & _ & _ & Hnt & _)]];
    [ | | rewrite Htupled in Hnt; discriminate Hnt ].
  - (* A full call: the tuple was the only argument. *)
    assert (rest_arity = []) as ->.
    { destruct rest_arity as [ | r0 rr]; [ reflexivity | exfalso ].
      rewrite length_app in Hfull. unfold arity_of_kinds in Hfull.
      rewrite length_map, repeat_length in Hfull.
      unfold cardinal_unarized in Hfull. rewrite Hsize in Hfull.
      simpl in Hfull. lia. }
    destruct rest as [ | r0 rest]; [ | discriminate Hlen_rest ].
    destruct rest' as [ | r0' rest']; [ | discriminate Hk_rest ].
    destruct ws' as [ | w2 ws'']; [ | discriminate Hlen_ws ].
    cbn [run_call] in Hflow' |- *.
    rewrite !app_nil_r in Hflow' |- *. rewrite (app_nil_r fields').
    revert Hflow'.
    match goal with
    | |- result_flow ?r -> _ =>
      destruct r as [s' | k' vs'' hp'' rcd'' | ] eqn:Hcc
    end;
      intros Hflow';
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & Hk1' & Hk2' & Hk3' & Hbind & ->).
    cbn [code_body code_params code_my_closure code_return_continuation
         code_exn_continuation code_params_arity code_result_arity]
      in Hk2', Hk3', Hbind.
    assert (Hlenp : length params = length (cd_params cd)).
    { rewrite Lp, Hsize, <- Hlen_f. symmetry.
      exact (bind_params_length _ _ _ _ Hbind). }
    destruct (bind_params_rel params fields fields'
                (repeat true (length params)) [(mc, Value)] [(mc, callee)]
                [(mc, callee')] callee_vars Henv0 Hlen_f' (repeat_length _ _)
                (Hparams Hlenp) Hbind)
      as (callee_vars' & Hbind' & Hok').
    rewrite !filter_kept_all in Hbind'.
    right. left. eexists. split.
    + eapply (call_code_rebuilt callee' id (Code params mc rk ek md cbody)
               _ _ _ _ _ _ _ _ _ Hun);
        cbn [code_params code_my_closure code_params_arity code_result_arity
             code_body].
      * apply rebuilt_code_no_statics; assumption.
      * rewrite Hkf, Hk1'. apply kinds_eqb_refl.
      * apply kinds_per_parameter_eqb_eq. exact Hk2'.
      * apply arity_eqb_eq. exact Hk3'.
      * exact Hbind'.
    + assert (Hway_e : way_out cont_rel over_rel [cd_exn cd] et et').
      { apply (way_out_transport _ _ [exn]); [ exact Hexn | reflexivity | ].
        intros i n n' Hn Hn'. destruct i as [ | [ | i]]; try discriminate Hn.
        injection Hn as <-. injection Hn' as <-.
        exact (read_transports _ _ W w1 _ Hnl_exn Hexnc
                 (Hexn_acc 0 w1 eq_refl) HWw1). }
      assert (Lret : length (cd_return cd) = length (unarize ret))
        by (rewrite Lr, <- Hk3'; reflexivity).
      assert (Htr_r : ∀ (i : nat) (n n' : node),
                nth_error returns i = Some n ->
                nth_error (cd_return cd) i = Some n' -> transports n' n).
      { intros i n n' Hn Hn'.
        assert (Hl1 : nth_error [w1] (pred (length [w1])) = Some w1)
          by reflexivity.
        exact (read_transports _ _ W w1 _ (result_not_local i)
                 (Hretc i n' Hn') (Hlast_ret i w1 n Hl1 Hn) HWw1). }
      destruct (proj1 Hret) as [Hret_w | Hret_r].
      * exact (code_entered id params mc rk ek md cbody cd callee fields
                 callee_vars callee_vars' rt rt' et et' statics statics' hp rcd
                   rcd'
                 Hun Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c
                 Hfresh_c Hne_k Hcounted Hbody Lr
                 (eq_trans Hk1' (unarize_per_parameter_eq _ _ Hk2')) Hbind Hok'
                 Hlive' (Hret_w _ Lret Htr_r) Hway_e).
      * apply (code_entered_ret id params mc rk ek md cbody cd callee fields
                 callee_vars callee_vars' rt rt' et et' statics statics' hp rcd
                   rcd'
                 Hun Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c
                 Hfresh_c
                 (proj1 (Hcode_rets id params mc rk ek md cbody Hexf
                           Hin_codes) Hne_k)
                 (proj2 (Hcode_rets id params mc rk ek md cbody Hexf
                           Hin_codes))
                 Hne_k Hcounted Hbody Lr
                 (eq_trans Hk1' (unarize_per_parameter_eq _ _ Hk2')) Hbind Hok'
                 Hlive'); [ | exact Hway_e ].
        rewrite <- Hk3'. exact (Hret_r _ Lret Htr_r).
  - (* An over-application: what the code returns is called with the
       arguments after the tuple. *)
    cbn [run_call] in Hflow' |- *.
    assert (Hcard : cardinal_unarized (cm_params_arity md) = length fields)
      by (unfold cardinal_unarized; rewrite Hsize; exact (eq_sym Hlen_f)).
    rewrite Hcard in Hflow' |- *.
    assert (E1 : firstn (length fields) (fields' ++ rest') = fields')
      by (rewrite <- Hlen_f'; apply firstn_length_app).
    assert (E2 : skipn (length fields) (fields' ++ rest') = rest')
      by (rewrite <- Hlen_f'; apply skipn_length_app).
    rewrite E1, E2, firstn_length_app, skipn_length_app.
    rewrite firstn_length_app, skipn_length_app in Hflow'.
    assert (Hskip : partially_apply
                      (arity_of_kinds (repeat Value (length fields))
                       ++ rest_arity)
                      (length (cm_params_arity md))
                    = rest_arity).
    { unfold partially_apply, arity_of_kinds.
      replace (length (cm_params_arity md))
        with (length (map Singleton (repeat Value (length fields))))
        by (rewrite length_map, repeat_length;
            first [ exact Hlen_f | exact (eq_sym Hlen_f) ]).
      apply skipn_length_app. }
    rewrite Hskip in Hflow' |- *.
    revert Hflow'.
    match goal with
    | |- result_flow ?r -> _ =>
      destruct r as [s' | k' vs'' hp'' rcd'' | ] eqn:Hcc
    end;
      intros Hflow';
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & Hk1' & Hk2' & _ & Hbind & ->).
    cbn [code_body code_params code_my_closure code_return_continuation
         code_exn_continuation code_params_arity code_result_arity]
      in Hk2', Hbind.
    (* The code returns one value. *)
    unfold partial_or_over_application in Hpoo.
    apply andb_true_iff in Hpoo as [_ Hpoo].
    rewrite (proj2 (Nat.ltb_lt _ _) Hover) in Hpoo. cbv iota in Hpoo.
    apply is_one_param_of_kind_value_eq in Hpoo.
    assert (Ecr : ∃ R0 : node, cd_return cd = [R0]).
    { rewrite Hpoo in Lr. simpl in Lr.
      destruct (cd_return cd) as [ | R0 [ | R1 rr]]; try discriminate Lr.
      exists R0. reflexivity. }
    destruct Ecr as (R0 & Ecr).
    assert (HR0 : nth_error (cd_return cd) 0 = Some R0)
      by (rewrite Ecr; reflexivity).
    (* There is an argument after the tuple, at a second witness. *)
    destruct rest_arity as [ | r_a rest_arity'].
    { exfalso. rewrite app_nil_r in Hover. unfold arity_of_kinds in Hover.
      rewrite length_map, repeat_length in Hover.
      unfold cardinal_unarized in Hover. rewrite Hsize in Hover. lia. }
    destruct ws' as [ | w2 ws'']; [ discriminate Hlen_ws | ].
    assert (Hhu_R0 : hu R0).
    { destruct (Hnext_acc 0 w1 w2 eq_refl eq_refl) as (vc & Hvc & Hw2v).
      pose proof (has_usage_accessor _ _ _ Hc _ _ _
                    (au_hu _ _ _ Hc _ Hau_called) (Hcalled_acc 1 w2 eq_refl))
        as Hhu_w2.
      pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_w2 Hw2v) as Hhu_vc.
      exact (read_hu _ _ _ _ _ (result_not_local 0) (Hretc 0 R0 HR0) Hvc
               HWw1 Hhu_vc). }
    (* The arguments after the tuple, at the witnesses after that of the
       code. *)
    assert (Hshift_R0 : ∀ (g : nat) (W' : node),
              wchain (from_node R0) g W' -> wchain (from_node N) (1 + g) W').
    { intros g W' HW'.
      exact (wchain_from_return _ 0 W R0 HW0 (Hretc 0 R0 HR0) g W' HW'). }
    assert (Hrem : args_rel (from_node R0) (r_a :: rest_arity') rest rest').
    { pose proof (args_rel_skipn _ _ _ _ _ 1 Ha Hshift_R0 ltac:(simpl; lia))
        as H.
      rewrite Hsizes in H. exact H. }
    assert (Hover_rel :
      over_rel R0
        (Over_application rest (r_a :: rest_arity') ret rt et)
        (Over_application rest' (r_a :: rest_arity') ret rt' et')).
    { apply (Over_rel R0 _ _ _ ret rt rt' et et' (w2 :: ws'') returns exn
               Hexf Hunt Hhu_R0).
      - discriminate.
      - exact Hlen_ws.
      - exact Hrem.
      - split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
        + intros g W' w HW' Hw.
          exact (Hcover (S g) W' w (Hshift_R0 g W' HW') Hw).
        + intros i w r Hw Hr. apply (Hlast_ret i w r); [ | exact Hr ].
          exact Hw.
        + intros g w Hw. exact (Hexn_acc (S g) w Hw).
        + intros g w w' Hw Hw'. exact (Hnext_acc (S g) w w' Hw Hw').
        + exists called_node. split; [ exact Hau_called | ].
          intros g w Hw. exact (Hcalled_acc (S g) w Hw).
        + intros W' w HW' Hw.
          exact (call_reads_hu _ _ _ _ Hreads0 1 W' w (Hshift_R0 0 W' HW')
                   Hw).
      - exact Hret.
      - exact Hexn. }
    (* The parameters get the fields of the tuple. *)
    assert (Hlenp : length params = length (cd_params cd)).
    { rewrite Lp, Hsize, <- Hlen_f. symmetry.
      exact (bind_params_length _ _ _ _ Hbind). }
    destruct (bind_params_rel params fields fields'
                (repeat true (length params)) [(mc, Value)] [(mc, callee)]
                [(mc, callee')] callee_vars Henv0 Hlen_f' (repeat_length _ _)
                (Hparams Hlenp) Hbind)
      as (callee_vars' & Hbind' & Hok').
    rewrite !filter_kept_all in Hbind'.
    right. left. eexists. split.
    + eapply (call_code_rebuilt callee' id (Code params mc rk ek md cbody)
               _ _ _ _ _ _ _ _ _ Hun);
        cbn [code_params code_my_closure code_params_arity code_result_arity
             code_body].
      * apply rebuilt_code_no_statics; assumption.
      * rewrite Hkf, Hk1'. apply kinds_eqb_refl.
      * apply kinds_per_parameter_eqb_eq. exact Hk2'.
      * apply arity_eqb_eq.
        destruct (code_fields_rebuild id (Code params mc rk ek md cbody) Hun)
          as (_ & _ & _ & _ & _ & Era & _).
        rewrite Era. reflexivity.
      * exact Hbind'.
    + apply (code_entered id params mc rk ek md cbody cd callee fields
               callee_vars callee_vars' _ _ et et' statics statics' hp rcd rcd'
               Hun Hflow' Hst Hscr_c Hcf_c Hraises_c Hclosed_c Hkinded_c
               Hfresh_c Hne_k Hcounted Hbody Lr
               (eq_trans Hk1' (unarize_per_parameter_eq _ _ Hk2')) Hbind Hok'
               Hlive').
      * rewrite Ecr. right. right. right. exists R0, R0, [].
        split; [ reflexivity | ].
        split; [ split; [ intros H; exact H | intros v0 v0' H; exact H ] | ].
        exact Hover_rel.
      * apply (way_out_transport _ _ [exn]); [ exact Hexn | reflexivity | ].
        intros i n n' Hn Hn'. destruct i as [ | [ | i]]; try discriminate Hn.
        injection Hn as <-. injection Hn' as <-.
        exact (read_transports _ _ W w1 _ Hnl_exn Hexnc
                 (Hexn_acc 0 w1 eq_refl) HWw1).
Qed.

(* A call of unknown arity, of a closure or of a partial application,
   returning to a way out whose values are answered. *)
Lemma called_of_returned (rt : cont_value) : returned_ok rt -> called_ok rt.
Proof.
  intros Hret_ok N callee callee' vs vs' a ret rt' et et' ws returns exn
    statics statics' hp rcd rcd' Hexf Hunt HhuN Hne Hlen_ws Hrel_c Hpa Ha
      Hvs_pa
    Hreads Hret Hrt_pa Hexn Hst Hcodes Hsflow Hflow'.
  unfold call_function in Hflow' |- *.
  destruct (shape_of_call statics callee Indirect vs a ret) as [shape | ]
    eqn:Hshape; [ | exact I ].
  destruct (partial_application_of callee)
    as [[[closure applied] aa] | ] eqn:Hpa_of.
  - (* A partial application: the call of its closure with the arguments
       that it keeps, then those of the call. *)
    destruct callee; try discriminate Hpa_of.
    injection Hpa_of as -> -> ->.
    unfold shape_of_call in Hshape. cbn [partial_application_of] in Hshape.
    destruct a as [ | a0 a1]; [ contradiction | ].
    destruct (kinds_eqb (map kind_of_value vs) (unarize (a0 :: a1))) eqn:Hk;
      [ | discriminate Hshape ].
    destruct Hrel_c
      as [_ (m & closure' & applied' & -> & _ & Hrel_cl & Hhum & _ & Hne_aa
             & Hend & Haa)].
    destruct Hpa as (Hun & Hcl_pa & Happ_pa).
    unfold shape_of_call. cbn [partial_application_of].
    rewrite (proj1 Ha), Hk.
    apply (closure_called rt rt' et et' closure closure' m N applied applied'
             vs vs' aa (a0 :: a1) ret ws returns exn statics statics' hp rcd
               rcd' shape
             Hret_ok Hexf Hunt Hne Hlen_ws Hrel_cl Hhum Hcl_pa).
    + apply (wchain_shift_end _ N (length aa) Hend).
      intros E. apply length_zero_iff_nil in E. contradiction.
    + exact Haa.
    + exact Ha.
    + right. apply kinds_eqb_eq. exact Hk.
    + intros id sl c Hof Hcode.
      destruct Hun as (id' & sl' & c' & Hof' & Hcode' & Hlt & Hnt).
      rewrite Hof in Hof'. injection Hof' as <- <-.
      rewrite Hcode in Hcode'. injection Hcode' as <-.
      exact (conj Hlt Hnt).
    + exact Hreads.
    + exact Hret.
    + exact Hrt_pa.
    + exact Hexn.
    + exact (every_app_intro _ _ _ Happ_pa Hvs_pa).
    + exact Hst.
    + exact Hcodes.
    + exact Hsflow.
    + exact Hshape.
    + exact Hflow'.
  - (* A closure. *)
    unfold shape_of_call in Hshape. rewrite Hpa_of in Hshape.
    assert (Hshape' :
      shape_of_closure_call statics callee Indirect vs a ret = Some shape)
      by exact Hshape.
    destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape')
      as (id0 & slots0 & c0 & vs0 & a0 & Hclosure0 & Hcode0 & _ & _ & _).
    destruct (rel_closure_of statics statics' callee callee' N id0 slots0 Hst
                Hrel_c HhuN Hclosure0)
      as (slots' & Hclosure').
    assert (Hpa' : partial_application_of callee' = None)
      by (destruct callee'; try discriminate Hclosure'; reflexivity).
    rewrite (shape_of_call_closure _ _ _ _ _ _ Hpa').
    destruct (code_is_tupled c0) eqn:Htupled0.
    { (* A closure of tupled code. *)
      exact (tupled_called rt rt' et et' callee callee' N vs vs' a ret ws
               returns exn statics statics' hp rcd rcd' shape id0 slots0 c0 Hexf
                 Hunt
               Hlen_ws Hrel_c HhuN Ha Hclosure0 Hcode0 Htupled0 Hreads Hret
               Hexn Hst Hcodes Hsflow Hshape' Hflow'). }
    apply (closure_called rt rt' et et' callee callee' N N [] [] vs vs' []
             a ret ws returns exn statics statics' hp rcd rcd' shape Hret_ok
               Hexf Hunt
             Hne Hlen_ws Hrel_c HhuN Hpa).
    + intros g W HW. exact HW.
    + split; [ reflexivity | exact I ].
    + exact Ha.
    + left. reflexivity.
    + intros id sl c Hof Hcode.
      rewrite Hclosure0 in Hof. injection Hof as <- <-.
      rewrite Hcode0 in Hcode. injection Hcode as <-.
      destruct (Hcodes id0 c0 Hcode0) as [Hin _].
      pose proof (Hunt id0 c0 Hin) as Hne_c. rewrite Htupled0 in Hne_c.
      split; [ | exact Htupled0 ].
      destruct (code_params_arity c0); [ contradiction | simpl; lia ].
    + exact Hreads.
    + exact Hret.
    + exact Hrt_pa.
    + exact Hexn.
    + exact Hvs_pa.
    + exact Hst.
    + exact Hcodes.
    + exact Hsflow.
    + exact Hshape'.
    + exact Hflow'.
Qed.

(* The values given to a way out, and the calls that return to one, are
   answered, by induction on the continuation: what is left of an
   over-application is a call that returns to what the
   over-application returns to. *)
Lemma answered_engine : ∀ c : cont_value, returned_ok c ∧ called_ok c.
Proof.
  intros c.
  assert (Hr : returned_ok c).
  { induction c as [h vars conts recursive | k | c IH extra
                   | remaining ra ret rt IHrt et IHet];
      apply returned_cases;
      intros n0 c' v v' statics statics' hp rcd rcd' Hover;
      destruct (over_rel_inv _ _ _ Hover)
        as (rem & rem' & ra' & ret' & rt0 & rt0' & et0 & et0' & ws & returns
            & exn & E & Ec' & Hrest);
      try discriminate E.
    intros Hrel_v Hpa_v Hc_pa Hst Hcodes Hsflow Hflow'.
    destruct Hrest as (Hexf & Hunt & Hhu0 & Hne & Hlen & Hargs & Hreads
                       & Hret & Hexn).
    subst c'. injection E as <- <- <- <- <-.
    inversion Hc_pa as [ | | | rem0 ra0 ret0 rt1 et1 Hrem_pa Hrt_pa Het_pa];
      subst.
    pose proof (called_of_returned rt IHrt) as Hcalled.
    exact (Hcalled n0 v v' remaining rem' ra ret rt0' et et0' ws returns exn
             statics statics' hp rcd rcd' Hexf Hunt Hhu0 Hne Hlen Hrel_v
               Hpa_v
             Hargs Hrem_pa Hreads Hret Hrt_pa Hexn Hst Hcodes Hsflow Hflow'). }
  split; [ exact Hr | exact (called_of_returned c Hr) ].
Qed.



Lemma apply_cont_answered (ac : apply_cont) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Apply_cont ac) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Apply_cont ac) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  related_intro. pose proof He as Hedges. destruct He as [Haliases Hnoted].
  unfold answered. rewrite step_apply_cont in Hflow' |- *.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs; [ | exact I ].
  change (rebuild en (Apply_cont ac)) with (rebuild_apply_cont sol en ac)
    in Ht.
  unfold rebuild_apply_cont in Ht.
  rewrite (rewrite_apply_cont_some en ac
             (args_have_sources vars _ vs Hvars Hnoted Hvs)) in Ht.
  subst e'.
  destruct (rewrite_simples_eval en vars vars' Henv (ac_args ac) vs Hnoted Hvs)
    as (vs' & Hvs' & Hkinds & Hrels).
  pose proof (apply_cont_args_at en vars vars' ce ac vs vs' Henv Hedges Hvs
                Hrels) as Hat.
  assert (Hk : ac_continuation ac
               ∈ free_continuations_of_expr
                   (Apply_cont (rebuilt_apply_cont en ac)))
    by (left; reflexivity).
  rewrite step_apply_cont. cbn [ac_args ac_continuation rebuilt_apply_cont].
  unfold apply_continuation in Hflow' |- *.
  destruct (Hconts _ Hk)
    as [(Hmnone & ns & Hns & Hway)
       | [(mask & c & c' & Hmask & Hc_ & Hc'_ & Hce & Hcrel & _)
         | (K & HK & m & ns & Hm & _ & Hns & Hrr)]].
  - (* No handler of the expression: a way out, which takes as many values
       as [ns]. *)
    rewrite Hmnone, Hvs'.
    destruct Hok as [_ [(bound & [Hb_sub _] & Hone) _]].
    cbn [exits_bound] in Hone.
    unfold exits_ac in Hone.
    assert (Eb : among (ac_continuation ac) bound = false).
    { destruct (among (ac_continuation ac) bound) eqn:Eb; [ | reflexivity ].
      pose proof (Hb_sub _ Eb) as H.
      rewrite (lookup_none_among _ _ Hmnone) in H. discriminate H. }
    rewrite Eb in Hone. cbn [orb] in Hone.
    unfold count_is in Hone. rewrite Hns in Hone.
    apply Nat.eqb_eq in Hone.
    pose proof Hflow as (ctx0 & ce0 & _ & _ & _ & _ & _ & _ & Hcodes
                         & Henv_pa & _ & Hconts_pa).
    simpl_state. rewrite Hheap.
    enough (Hres :
      result_rel
        (apply_cont_value (resolve (ac_continuation ac) conts) vs statics hp
          rcd)
        (apply_cont_value (resolve (ac_continuation ac) conts') vs' statics'
           hp rcd')).
    { revert Hres. unfold result_rel.
      destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs statics
                  hp rcd) as [s' | k' vs0 hp0 rcd0 | ];
        [ | intros H; exact H | intros _; exact I ].
      intros [H | [(t' & Ht & Hr) | (t1 & t' & n & Ht & Hrun & Hr)]].
      - left. exact H.
      - right. left. exists t'. split; assumption.
      - right. right. right. exists t', n.
        rewrite (run_succ_step (S n) _ t1); [ split; assumption | ].
        rewrite step_apply_cont.
        cbn [ac_args ac_continuation rebuilt_apply_cont].
        rewrite Hmnone, Hvs'. unfold apply_continuation. exact Ht. }
    apply (proj1 (answered_engine (resolve (ac_continuation ac) conts)) _ ns
             vs vs' statics statics' hp rcd rcd' Hway).
    + split; [ rewrite (eval_simples_length _ _ _ Hvs); symmetry; exact Hone
             | split; [ exact Hkinds | ] ].
      intros i n v v' Hn Hv Hv' Hu. exact (Hat ns i n v v' Hns Hn Hv Hv' Hu).
    + exact Hst.
    + exact Hcodes.
    + exact Hsflow.
    + exact (simples_pa_ok _ _ _ _ Henv_pa Hvs).
    + unfold resolve.
      destruct (lookup (ac_continuation ac) conts) as [c0 | ] eqn:Hl.
      * exact (Hconts_pa _ c0 (lookup_in _ _ _ Hl)).
      * apply Outer_pa.
    + exact (result_flow_of_ok _ Hflow').
  - (* A handler of the expression. *)
    unfold resolve in Hflow' |- *.
    rewrite Hmask, (eval_simples_filter_kept vars' mask _ _ Hvs'), Hc_, Hc'_.
    rewrite Hc_ in Hflow'.
    assert (Hargs : ∀ (i : nat) (p : variable) (v v' : value),
              nth_error (cont_params c) i = Some (Node_variable p) ->
              nth_error vs i = Some v -> nth_error vs' i = Some v' ->
              is_var_used sol p (kind_of_value v) = true ->
              rel v (Node_variable p) v').
    { intros i p v v' Hp Hv Hv' Hu.
      apply (Hat _ i _ v v' Hce Hp Hv Hv').
      destruct (is_var_used_cases p _ Hu) as [H | H];
        [ left; exact H | right; exact H ]. }
    pose proof (handler_entered mask c c' vs vs' statics statics' hp rcd rcd'
      Hcrel
                  Hst Hkinds Hargs) as Hent.
    rewrite Hheap.
    revert Hflow' Hent.
    destruct (apply_cont_value c vs statics hp rcd) as [s' | | ];
      intros Hflow' Hent; [ | contradiction | exact I ].
    right. left. destruct (Hent Hflow') as (_ & t' & Hstep & Hrel').
    exists t'. split; [ exact Hstep | exact Hrel' ].
  - (* The return continuation of code that changes its calling
       convention: the results that the code keeps. *)
    assert (Ev : eval_simples vars'
                   match lookup (ac_continuation ac) (env_conts en) with
                   | Some mask =>
                     filter_kept mask
                       (map (rewrite_simple sol en) (ac_args ac))
                   | None => map (rewrite_simple sol en) (ac_args ac)
                   end
                 = Some (filter_kept m vs')).
    { destruct Hm as [Hm | [Hm Hall]]; rewrite Hm;
        [ exact (eval_simples_filter_kept vars' m _ _ Hvs') | ].
      rewrite (filter_kept_true m vs' Hall). exact Hvs'. }
    rewrite Ev.
    destruct Hok as [_ [(bound & [_ Hb_rets] & Hone) (_ & _ & _ & Hnr)]].
    cbn [exits_bound] in Hone. unfold exits_ac in Hone.
    rewrite (Hb_rets _ _ HK) in Hone. cbn [orb] in Hone.
    unfold count_is in Hone. rewrite Hns in Hone.
    apply Nat.eqb_eq in Hone.
    pose proof Hflow as (ctx0 & ce0 & _ & _ & _ & _ & _ & _ & Hcodes
                         & Henv_pa & _ & Hconts_pa).
    simpl_state. rewrite Hheap.
    enough (Hres :
      result_rel
        (apply_cont_value (resolve (ac_continuation ac) conts) vs statics hp
          rcd)
        (apply_cont_value (resolve (ac_continuation ac) conts')
           (filter_kept m vs') statics' hp rcd')).
    { revert Hres. unfold result_rel.
      destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs statics
                  hp rcd) as [s' | k' vs0 hp0 rcd0 | ];
        [ | intros H; exact H | intros _; exact I ].
      intros [H | [(t' & Ht & Hr) | (t1 & t' & n & Ht & Hrun & Hr)]].
      - left. exact H.
      - right. left. exists t'. split; assumption.
      - right. right. right. exists t', n.
        rewrite (run_succ_step (S n) _ t1); [ split; assumption | ].
        rewrite step_apply_cont.
        cbn [ac_args ac_continuation rebuilt_apply_cont].
        rewrite Ev.
        unfold apply_continuation. exact Ht. }
    apply (ret_answered m K ns _ _ vs vs' statics statics' hp rcd rcd' Hrr
             (proj1 (answered_engine _))).
    + destruct (Hnr _ _ HK) as [_ H].
      cbn [ret_calls_ok] in H. rewrite Pos.eqb_refl in H.
      cbn [negb orb] in H.
      exact (simples_of_kinds_eval _ vars vars' Henv _ K vs H Hvs).
    + split; [ rewrite (eval_simples_length _ _ _ Hvs); symmetry; exact Hone
             | split; [ exact Hkinds | ] ].
      intros i n v v' Hn Hv Hv' Hu. exact (Hat ns i n v v' Hns Hn Hv Hv' Hu).
    + exact Hst.
    + exact Hcodes.
    + exact Hsflow.
    + exact (simples_pa_ok _ _ _ _ Henv_pa Hvs).
    + unfold resolve.
      destruct (lookup (ac_continuation ac) conts) as [c0 | ] eqn:Hl.
      * exact (Hconts_pa _ c0 (lookup_in _ _ _ Hl)).
      * apply Outer_pa.
    + exact (result_flow_of_ok _ Hflow').
Qed.

(* A call of unknown arity whose arguments are not of the kinds of its
   arity goes wrong. *)
Lemma call_function_kinds (callee : value) (vs : list value) (a ret : arity)
    (rt et : cont_value) (statics : static_env) (hp : heap)
    (rcd : read_record) :
  indirect_codes ->
  codes_ok is_local G noted deps cds keep codes symbols False statics ->
  map kind_of_value vs <> unarize a ->
  call_function callee Indirect vs a ret rt et statics hp rcd = Undefined.
Proof.
  intros Hunt Hcodes Hk. unfold call_function.
  destruct (shape_of_call statics callee Indirect vs a ret) as [shape | ]
    eqn:Hshape; [ | reflexivity ].
  unfold shape_of_call in Hshape.
  destruct (partial_application_of callee) as [[[cl app] aa] | ].
  - exfalso. destruct a as [ | a0 a1]; [ discriminate Hshape | ].
    destruct (kinds_eqb (map kind_of_value vs) (unarize (a0 :: a1))) eqn:E;
      [ | discriminate Hshape ].
    apply kinds_eqb_eq in E. exact (Hk E).
  - destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
      as (id & sl & c & vs0 & a0 & Hcl & Hcode & _ & Hafc & Hcall).
    destruct (code_is_tupled c) eqn:Hnt.
    { exfalso. unfold arguments_for_code in Hafc. rewrite Hnt in Hafc.
      unfold detuple in Hafc.
      destruct vs as [ | v0 vs]; [ discriminate Hafc | ].
      destruct a as [ | b0 b1]; [ discriminate Hafc | ].
      destruct (kinds_eqb (map kind_of_value (v0 :: vs)) (unarize (b0 :: b1)))
        eqn:E; [ | discriminate Hafc ].
      apply kinds_eqb_eq in E. exact (Hk E). }
    rewrite (arguments_for_code_not_tupled _ _ _ _ Hnt) in Hafc.
    injection Hafc as <- <-.
    destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
      as [[_ ->] | [(_ & Hk' & _) | (_ & Hk' & _)]];
      [ | destruct (Hk Hk') | destruct (Hk Hk') ].
    cbn [run_call]. unfold call_code.
    destruct (binds_statics (code_body c)); [ reflexivity | ].
    destruct (kinds_eqb (map kind_of_value vs) (unarize a)) eqn:E;
      [ apply kinds_eqb_eq in E; destruct (Hk E) | reflexivity ].
Qed.

(* A call that returns to a handler does not exit. *)
Lemma call_function_closure_not_exit (callee : value) (vs : list value)
    (a ret : arity) (h : continuation_handler) (hvars : env)
    (hconts : cont_env) (recursive : list (continuation * continuation_handler))
    (et : cont_value) (statics : static_env) (hp : heap) (rcd : read_record)
    (k : continuation)
    (vs' : list value) (hp' : heap) (rcd0 : read_record) :
  call_function callee Indirect vs a ret (Closure h hvars hconts recursive) et
    statics hp rcd
  <> Exit k vs' hp' rcd0.
Proof.
  unfold call_function.
  destruct (shape_of_call statics callee Indirect vs a ret)
    as [[cl c vs0 a0 | cl c vs0 rem a0 ra | partial] | ];
    cbn [run_call].
  - apply call_code_not_exit.
  - apply call_code_not_exit.
  - destruct h as [params md body]. cbn [apply_cont_value].
    destruct (kinds_eqb _ _); [ | discriminate ].
    destruct (bind_params _ _ _); discriminate.
  - discriminate.
Qed.

(* A call whose return continuation does not exit when given one value
   does not exit. *)
Lemma call_function_not_exit (callee : value) (vs : list value)
    (a ret : arity) (rt et : cont_value) (statics : static_env) (hp : heap)
    (rcd : read_record)
    (k : continuation) (vs' : list value) (hp' : heap) (rcd0 : read_record) :
  (ret = [Singleton Value] ->
   ∀ v : value, apply_cont_value rt [v] statics hp rcd <> Exit k vs' hp' rcd0)
     ->
  call_function callee Indirect vs a ret rt et statics hp rcd
  <> Exit k vs' hp' rcd0.
Proof.
  intros H. unfold call_function.
  pose proof (shape_of_call_arities statics callee Indirect vs a ret) as Ha.
  destruct (shape_of_call statics callee Indirect vs a ret)
    as [[cl c vs0 a0 | cl c vs0 rem a0 ra | partial] | ];
    cbn [run_call].
  - apply call_code_not_exit.
  - apply call_code_not_exit.
  - exact (H (proj1 Ha) partial).
  - discriminate.
Qed.

(* A handler that code returns to is a closure. *)
Lemma cont_rel_closure (mask : list bool) (c c' : cont_value) :
  cont_rel mask c c' ->
  ∃ (h : continuation_handler) (hvars : env) (hconts : cont_env)
    (recursive : list (continuation * continuation_handler)),
    c = Closure h hvars hconts recursive.
Proof. intros H. destruct H; eexists _, _, _, _; reflexivity. Qed.

(* A call of unknown arity, of the generic application, is answered:
   the arguments that the rebuild keeps are related at the parameters
   that have a use of the witnesses that their groups reach from what the
   callee may come from ([unknown_mask_used]), the witnesses of the call
   stand for those ([wchain_cover]), and the call is answered whatever
   its shape ([answered_engine]), after the binding of the wrapper if the
   rebuild gives the call one. *)
Lemma indirect_answered (ap : apply_expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (rcd rcd' : read_record)
    (e' : expr) :
  ap_call_kind ap = Indirect ->
  related (State (Apply ap) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Apply ap) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  intros Hkind. related_intro.
  destruct He as (Hfrag_ap & Hnoted_callee & Hnoted_args & returns & exn
                  & Hret_ce & Hexn_ce & Hcall).
  destruct Hok as (Hlive & (bound & [Hb_sub Hb_rets] & Hexits) & _ & Hcf
                   & (Hfresh & (ks & Hks & Hagree) & xs & Hxs & Hxagree)
                   & Hnr_e).
  (* A call of unknown arity returns to no return continuation of code
     that changes. *)
  assert (Hacc0 : apply_calling_convention_change F sol cannot_change
                    code_deps unit_code_ids code_metadatas ap
                  = Not_changing_calling_convention)
    by (unfold apply_calling_convention_change, code_id_actually_called;
        rewrite Hkind; reflexivity).
  assert (Hnr_ap :
    ∀ (r : continuation) (K : list flambda_kind),
      lookup r rets = Some K ->
      ap_exn_continuation ap <> r
      ∧ (ap_continuation ap = r -> unarize (ap_return_arity ap) = K)).
  { intros r K Hr. destruct (Hnr_e r K Hr) as [Hr1 Hr2]. split.
    - intros E. cbn [no_raise_to] in Hr1. rewrite E, Pos.eqb_refl in Hr1.
      discriminate Hr1.
    - intros E. pose proof Hr2 as H. cbn [ret_calls_ok] in H.
      rewrite E, Pos.eqb_refl in H. cbn [negb orb] in H.
      apply kinds_eqb_eq in H. symmetry. exact H. }
  cbn [exits_bound] in Hexits. cbn [checked] in Hcf.
  cbn [calls_kinded] in Hks. cbn [raises_kept] in Hxs.
  apply andb_true_iff in Hexits as [Hk_exn Hk_ret].
  apply orb_true_iff in Hk_exn, Hk_ret.
  assert (Hcount_rets : ∀ (r : continuation) (K : list flambda_kind),
            lookup r rets = Some K -> ap_continuation ap = r ->
            count_is ce r (length (unarize (ap_return_arity ap))) = true).
  { intros r K Hr E. subst r. destruct Hk_ret as [H | H]; [ | exact H ].
    rewrite (Hb_rets _ K Hr) in H. discriminate H. }
  assert (Hk_exn' : among (ap_exn_continuation ap) (map fst (env_conts en))
                    || count_is ce (ap_exn_continuation ap)
                         (S (length (ap_exn_extra_args ap)))
                    = true).
  { apply orb_true_iff. destruct Hk_exn as [H | H];
      [ left; exact (Hb_sub _ H) | right; exact H ]. }
  assert (Hk_ret' : among (ap_continuation ap) (map fst (env_conts en))
                    || count_is ce (ap_continuation ap)
                         (length (unarize (ap_return_arity ap)))
                    = true).
  { apply orb_true_iff. destruct Hk_ret as [H | H];
      [ left; exact (Hb_sub _ H) | right; exact H ]. }
  clear Hk_exn Hk_ret. rename Hk_exn' into Hk_exn. rename Hk_ret' into Hk_ret.
  assert (Hff_exn : ap_exn_continuation ap <> first_fresh).
  { intros E. apply not_true_iff_false in Hfresh. apply Hfresh.
    apply among_In. cbn [continuations_of continuations_in_apply].
    right. left. exact E. }
  apply negb_true_iff in Hcf as Hnex.
  assert (Hunt : indirect_codes).
  { intros id c Hin.
    pose proof Hfrag_ap as Hall. unfold apply_in_flow_fragment in Hall.
    rewrite Hkind, forallb_forall in Hall.
    specialize (Hall (id, c) Hin). cbv beta iota in Hall.
    destruct (code_is_tupled c); [ exact Hall | ].
    apply negb_true_iff in Hall.
    intros E. rewrite E in Hall. discriminate Hall. }
  unfold call_edges in Hcall. rewrite Hkind in Hcall.
  destruct Hcall as (ws & apply_node & called_node & Hlen_ws & Hch & Hused
                     & Hcond & Happly).
  pose proof Hflow as (ctx0 & ce0 & _ & _ & _ & _ & _ & _ & Hcodes & Henv_pa
                       & _ & Hconts_pa).
  (* What the call reads: the first witness of the call reads the call
     witness of unknown arity of the callee. *)
  assert (Hw0 : ∃ w0 : node, hd_error ws = Some w0).
  { destruct ws as [ | w0 ws']; [ | eexists; reflexivity ].
    unfold call_groups, at_least_one_group in Hlen_ws.
    destruct (group_by_parameter _ _); discriminate Hlen_ws. }
  destruct Hw0 as (w0 & Hw0).
  assert (Hgrp0 : ∃ grp0 : list simple,
            nth_error (combine (call_groups ap) ws) 0 = Some (grp0, w0)).
  { destruct ws as [ | w0' ws']; [ discriminate Hw0 | ].
    simpl in Hw0. injection Hw0 as <-.
    unfold call_groups, at_least_one_group.
    destruct (group_by_parameter _ _) as [ | grp0 grps];
      [ exists []; reflexivity | exists grp0; reflexivity ]. }
  destruct Hgrp0 as (grp0 & Hgrp0).
  destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch Hgrp0)
    as (_ & _ & Hcalled0 & _).
  pose proof (au_hu _ _ _ Hc _ (any_usage_input _ _ _ Hc _ Hused))
    as Hhu_called.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_called Hcalled0)
    as Hhu_w0.
  pose proof (cond_alias_alias is_local G sol Hc ctx w0 apply_node Hlive
                (Hcond w0 Hw0)) as Hal_wa.
  pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_w0 Hal_wa) as Hhu_a.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu_a Happly) as Hhu_callee.
  (* The step of the first run. *)
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | unfold answered; rewrite step_apply, Hcallee; exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
    [ | unfold answered; rewrite step_apply, Hcallee, Hvs; exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
    eqn:Hextra;
    [ | unfold answered; rewrite step_apply, Hcallee, Hvs, Hextra; exact I ].
  assert (Hstep1 :
    step (State (Apply ap) vars conts statics hp rcd)
    = call_function callee Indirect vs (ap_args_arity ap) (ap_return_arity ap)
        (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rcd)
    by (rewrite step_apply, Hcallee, Hvs, Hextra, Hkind; reflexivity).
  rewrite Hstep1 in Hflow'.
  (* A call without argument goes wrong. *)
  assert (Hne_or : ap_args_arity ap <> [] ∨ ap_args_arity ap = []).
  { destruct (ap_args_arity ap); [ right; reflexivity | left; discriminate ].
  }
  destruct Hne_or as [Hne_a | Hnil].
  2: { unfold answered. rewrite Hstep1, Hnil. unfold call_function.
       destruct (shape_of_call statics callee Indirect vs []
                   (ap_return_arity ap)) as [shape | ] eqn:Hshape;
         [ exfalso | exact I ].
       unfold shape_of_call in Hshape.
       destruct (partial_application_of callee) as [[[cl app] aa] | ];
         [ discriminate Hshape | ].
       destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
         as (id & sl & c & vs0 & a0 & Hcl & Hcode & _ & Hafc & Hcall).
       destruct (Hcodes id c Hcode) as [Hin _].
       destruct (code_is_tupled c) eqn:Hnt.
       { unfold arguments_for_code in Hafc. rewrite Hnt in Hafc.
         unfold detuple in Hafc. destruct vs; discriminate Hafc. }
       pose proof (Hunt id c Hin) as Hne_c. rewrite Hnt in Hne_c.
       rewrite (arguments_for_code_not_tupled _ _ _ _ Hnt) in Hafc.
       injection Hafc as <- <-.
       destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
         as [[Hfull _] | [(Hover & _) | (_ & _ & Hpoo & _)]].
       - simpl in Hfull. destruct (code_params_arity c);
           [ contradiction | discriminate Hfull ].
       - simpl in Hover. lia.
       - unfold partial_or_over_application in Hpoo.
         apply andb_true_iff in Hpoo as [_ Hpoo]. simpl in Hpoo.
         discriminate Hpoo. }
  (* A constant callee is no closure. *)
  assert (Hnc_or : (∀ c0 : const, ap_callee ap <> Const c0)
                   ∨ ∃ c0 : const, callee = Value_const c0).
  { destruct (ap_callee ap) as [x | sym | c0];
      [ left; intros c0 E; discriminate E | left; intros c0 E; discriminate E
      | right ].
    exists c0. simpl in Hcallee. injection Hcallee as <-. reflexivity. }
  destruct Hnc_or as [Hnc | (c0 & ->)].
  2: { unfold answered. rewrite Hstep1. exact I. }
  (* The extra arguments of the exception, in the second run. *)
  destruct (rewrite_simples_kinds en vars vars' Henv _ extra Hextra)
    as (extra' & Hextra' & Hkx).
  (* The rebuilt call. *)
  assert (Hdead : (simple_is_dead sol en (ap_callee ap)
                   || existsb (simple_is_dead sol en) (ap_args ap)) = false).
  { apply orb_false_iff. split.
    - exact (simple_not_dead en vars vars' _ _ Hvars Hnoted_callee Hcallee).
    - apply not_true_iff_false. intros Hex.
      apply existsb_exists in Hex as (s & Hin & Hd).
      destruct (eval_simples_member vars _ vs s Hvs Hin) as (w & Hw).
      rewrite (simple_not_dead en vars vars' s w Hvars (Hnoted_args s Hin) Hw)
        in Hd.
      discriminate Hd. }
  subst e'. cbn [rebuild_expr_cc] in Hconts |- *.
  rewrite (rebuild_apply_indirect en ap Hkind) in Hconts |- *.
  unfold rebuild_apply in Hconts |- *. rewrite Hdead in Hconts |- *.
  subst hp'.
  destruct (rewrite_simple_eval en vars vars' (ap_callee ap) callee Henv
              Hnoted_callee Hcallee)
    as (callee' & Hcallee' & _ & Hkept_c).
  destruct (Hkept_c (kept_of_hu en _ Hhu_callee)) as [Hrw_c Hrel_c].
  rewrite Hrw_c in Hcallee'.
  set (N := node_of_simple (ap_callee ap)) in *.
  (* The chain of the call. *)
  destruct (indirect_pending is_local G sol noted cds keep no_loc Hc Hlocal
      Hconsts
              Hsyms ctx ap vars vs returns exn ws apply_node called_node Hlive
              Hvars Hnoted_args Hvs Hne_a Hlen_ws Hch Hused Hcond Happly)
    as (Hlen_p & Hchain & _ & _ & _ & _).
  assert (Hcover : ∀ (g : nat) (W w : node),
            wchain (from_node N) g W -> nth_error ws g = Some w ->
            may_come_from sol W w).
  { intros g W w HW Hw.
    apply (wchain_cover g (fun y => may_come_from sol y N) ws W w Hchain);
      [ | exact Hw ].
    apply (wchain_mono (from_node N)); [ intros y [Hy _]; exact Hy | ].
    exact HW. }
  assert (Hcg : call_groups ap
                = group_by_sizes (arity_sizes (ap_args_arity ap)) (ap_args ap))
    by (unfold call_groups; exact (group_by_parameter_groups _ _ Hne_a)).
  assert (Hgrp_of : ∀ (g : nat) (w : node),
            nth_error ws g = Some w ->
            ∃ grp : list simple,
              nth_error (combine (call_groups ap) ws) g = Some (grp, w)).
  { intros g w Hw. destruct (nth_error (call_groups ap) g) as [grp | ] eqn:Hg.
    - exists grp. exact (nth_error_combine_eq _ _ _ _ _ Hg Hw).
    - exfalso. apply nth_error_None in Hg.
      assert (g < length ws) by (apply nth_error_Some; congruence). lia. }
  (* What the call reads. *)
  assert (Hreads : call_reads (from_node N) ws returns exn).
  { split; [ exact Hcover | ].
    split.
    { intros i w r Hw Hr. destruct (Hgrp_of _ w Hw) as (grp & HL).
      destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch HL)
        as (_ & _ & _ & Hlast & _).
      apply Hlast; [ | exact Hr ].
      assert (Hlw : pred (length ws) < length ws)
        by (apply nth_error_Some; congruence).
      rewrite length_combine, <- Hlen_ws, Nat.min_id. lia. }
    split.
    { intros g w Hw. destruct (Hgrp_of _ w Hw) as (grp & HL).
      destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch HL) as (_ & Hx & _).
      exact Hx. }
    split.
    { intros g w w' Hw Hw'. destruct (Hgrp_of _ w Hw) as (grp & HL).
      destruct (Hgrp_of _ w' Hw') as (grp' & HL').
      destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch HL)
        as (_ & _ & _ & _ & Hnext).
      exact (Hnext grp' w' HL'). }
    split.
    { exists called_node.
      split; [ exact (any_usage_input _ _ _ Hc _ Hused) | ].
      intros g w Hw. destruct (Hgrp_of _ w Hw) as (grp & HL).
      destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch HL) as (_ & _ & Hx & _).
      exact Hx. }
    (* The witness of the first group reads the callee, as the call. *)
    intros W w (y & [Hy _] & Hcons & _) Hw.
    assert (Hhd : hd_error ws = Some w)
      by (destruct ws as [ | w1 ws1]; [ discriminate Hw | exact Hw ]).
    pose proof (cond_alias_alias is_local G sol Hc ctx w apply_node Hlive
                  (Hcond w Hhd)) as Hal.
    destruct (Hgrp_of 0 w Hw) as (grp & HL).
    destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch HL) as (_ & _ & Hx & _).
    pose proof (has_usage_accessor _ _ _ Hc _ _ _
                  (au_hu _ _ _ Hc _ (any_usage_input _ _ _ Hc _ Hused)) Hx)
      as Hhu_wfst.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hhu_wfst Hal) as Hhu_ap.
    exact (read_hu y N apply_node W _ call_witness_unknown_not_local Hcons
             Happly Hy Hhu_ap). }
  assert (Hlen_wa : length ws = length (ap_args_arity ap)).
  { rewrite Hlen_ws, Hcg, length_group_by_sizes, length_arity_sizes.
    reflexivity. }
  (* The arguments of the second run. *)
  pose proof (arguments_used_lengths Unknown_arity_code_pointer
                (arity_sizes (ap_args_arity ap))
                (get_direct_sources sol N))
    as Hms.
  assert (Hargs_eq :
    rebuild_arguments F sol en ap
    = mask_arguments en
        (concat
           (arguments_used_by_call F sol Unknown_arity_code_pointer
              (get_direct_sources sol N) (arity_sizes (ap_args_arity ap))))
        (ap_args ap)).
  { unfold rebuild_arguments. rewrite Hkind, <- unknown_mask_eq.
    unfold N. destruct (ap_callee ap) as [x | sym | c0];
      [ reflexivity | reflexivity | destruct (Hnc c0 eq_refl) ]. }
  set (mask := concat
                 (arguments_used_by_call F sol Unknown_arity_code_pointer
                    (get_direct_sources sol N)
                    (arity_sizes (ap_args_arity ap)))) in *.
  assert (Hlen_or :
    length (ap_args ap) = list_sum (arity_sizes (ap_args_arity ap))
    ∨ map kind_of_value vs <> unarize (ap_args_arity ap)).
  { destruct (Nat.eq_dec (length (ap_args ap))
                (list_sum (arity_sizes (ap_args_arity ap)))) as [E | E];
      [ left; exact E | right ].
    intros Hk. apply E. rewrite sum_arity_sizes, <- Hk, length_map.
    symmetry. exact (eval_simples_length _ _ _ Hvs). }
  destruct Hlen_or as [Hlen_args | Hkbad].
  2: { unfold answered. rewrite Hstep1.
       rewrite (call_function_kinds _ _ _ _ _ _ _ _ _ Hunt Hcodes Hkbad).
       exact I. }
  assert (Hmlen : length mask = length (ap_args ap)).
  { unfold mask. rewrite (concat_length_sizes _ _ Hms). symmetry.
    exact Hlen_args. }
  (* The callee, as the mask reads it. *)
  assert (Hcov : src_covers (get_direct_sources sol N) N).
  { unfold get_direct_sources.
    destruct (s_any_source sol N) eqn:E; cbn [src_covers]; [ exact E | ].
    intros x Hx. left. apply andb_true_iff in Hx as [Hx _]. exact Hx. }
  assert (Hread : ∀ (grp : list simple) (c0 : node),
            hd_error (combine (call_groups ap) ws) = Some (grp, c0) ->
            ∃ a : node,
              g_accessor G a (Field_call_witness Unknown_arity_code_pointer) N
              ∧ (a = c0 ∨ s_alias sol c0 a = true)).
  { intros grp c0 Hhd. exists apply_node. split; [ exact Happly | right ].
    exact (cond_alias_alias is_local G sol Hc ctx c0 apply_node Hlive
             (Hcond c0 (hd_error_combine _ _ _ _ Hhd))). }
  (* The group of the argument at [i], and the witness of the call of
     that group. *)
  assert (Hgroup : ∀ (i : nat) (s : simple),
            nth_error (ap_args ap) i = Some s ->
            ∃ (g k : nat) (grp : list simple) (w : node),
              group_position (arity_sizes (ap_args_arity ap)) i = Some (g, k)
              ∧ nth_error grp k = Some s
              ∧ nth_error (combine (call_groups ap) ws) g = Some (grp, w)
              ∧ nth_error ws g = Some w).
  { intros i s Hs.
    assert (Hi : i < list_sum (arity_sizes (ap_args_arity ap))).
    { rewrite <- Hlen_args. apply nth_error_Some. congruence. }
    destruct (group_position_some _ _ Hi) as (g & k & Hpos).
    destruct (group_by_sizes_nth _ _ _ _ _ _ Hpos Hs) as (grp & Hgrp & Hs').
    destruct (group_position_lt _ _ _ _ Hpos) as [Hg _].
    rewrite length_arity_sizes in Hg.
    destruct (nth_error ws g) as [w | ] eqn:Hw;
      [ | apply nth_error_None in Hw; lia ].
    exists g, k, grp, w. split; [ exact Hpos | split; [ exact Hs' | ] ].
    split; [ | exact Hw ].
    apply nth_error_combine_eq; [ rewrite Hcg; exact Hgrp | exact Hw ]. }
  destruct (mask_arguments_eval en vars vars' Henv mask (ap_args ap) vs
              Hnoted_args Hvs Hmlen)
    as (vs' & Hvs' & Hkinds' & Hrels').
  { intros i s Hs Hm.
    destruct (Hgroup i s Hs) as (g & k & grp & w & Hpos & Hs' & HLg & _).
    destruct (concat_group_nth _ _ Hms i g k Hpos)
      as (m & Hmg & _ & Hnth).
    unfold mask in Hm. rewrite Hnth in Hm.
    exact (unknown_mask_kept returns exn called_node _ _ _ _ Hch Hcov Hread
             g k grp w m s HLg Hmg Hs' Hm). }
  (* The arguments, at the witnesses that their groups reach from what
     the callee may come from. *)
  assert (Ha_rel : args_rel (from_node N) (ap_args_arity ap) vs vs').
  { apply args_rel_intro; [ exact Hkinds' | ].
    intros i v v' g k W p Hv Hv' Hpos HW Hp Hhu_p.
    destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
    destruct (Hgroup i s Hs) as (g' & k' & grp & w & Hpos' & Hs' & HLg & Hw).
    rewrite Hpos in Hpos'. injection Hpos' as <- <-.
    destruct (caller_chain_nth G _ _ _ _ _ _ _ Hch HLg) as (Hargs_w & _).
    destruct (concat_group_nth _ _ Hms i g k Hpos)
      as (m & Hmg & Hkm & Hnth).
    assert (Hm : nth_error mask i = Some true).
    { unfold mask. rewrite Hnth.
      refine (unknown_mask_used _ _ _ _ g k W p m HW Hmg Hp Hhu_p Hkm).
      intros y [Hy Hyy]. unfold get_direct_sources.
      destruct (s_any_source sol N) eqn:E; cbn [src_has]; [ exact I | ].
      destruct Hy as (_ & [Has | Hsy] & _);
        [ rewrite Has in E; discriminate E | ].
      rewrite Hsy, (proj1 Hyy). reflexivity. }
    pose proof (Hrels' i s v v' Hs Hv Hv' Hm) as Hrel_s.
    pose proof (flows_hs is_local G sol keep called no_loc Hc _ _
                  (simple_flows is_local G sol noted cds keep no_loc Hc Hconsts
                      Hsyms
                     _ _ _ Hvars (Hnoted_args s (nth_error_In _ _ Hs)) Hsv))
      as Hhs.
    exact (rel_argument W w p (node_of_simple s) k v v'
             (Hcover g W w HW Hw) (Hargs_w k s Hs') Hp Hhs Hrel_s). }
  (* The exception continuation of the call: a way out, or a handler in
     scope that keeps the exception, with the extra arguments. *)
  assert (Hexn_to :
    ∃ extra'' : list value,
      eval_simples vars' (map fst (rebuild_exn_extra_args sol en ap))
      = Some extra''
      ∧ way_out cont_rel over_rel [exn]
          (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
          (with_extra_args (resolve (ap_exn_continuation ap) conts')
             extra'')).
  { pose proof (Hxagree (ap_exn_continuation ap)) as Hxa.
    assert (Hexn_free : ap_exn_continuation ap
                        ∈ free_continuations_of_expr
                            (make_apply_wrapper first_fresh en
                               (rebuilt_call en ap)))
      by exact (proj2 (make_apply_wrapper_free en (rebuilt_call en ap)
                         Hff_exn)).
    unfold rebuild_exn_extra_args.
    destruct (Hconts _ Hexn_free)
      as [(Hm_exn & nse & Hnse & Hway_exn)
         | [(mask1 & c1 & c1' & Hmask1 & Hc1 & Hc1' & Hce1 & Hcrel1 & _)
           | (Kx & HKx & _)]].
    3: { exfalso. exact (proj1 (Hnr_ap _ _ HKx) eq_refl). }
    - (* A way out: no extra argument. *)
      rewrite Hm_exn in Hxa |- *.
      destruct (lookup (ap_exn_continuation ap) xs) as [bx | ] eqn:Hxl;
        [ simpl in Hxa; destruct Hxa | ].
      assert (Hnox : ap_exn_extra_args ap = []).
      { unfold no_exn_extra_args in Hxs.
        destruct (ap_exn_extra_args ap); [ reflexivity | discriminate Hxs ]. }
      unfold exn_extra_simples in Hextra, Hextra'.
      rewrite Hnox in Hextra, Hextra' |- *.
      cbn [map eval_simples] in Hextra, Hextra' |- *.
      injection Hextra as <-. injection Hextra' as <-.
      exists []. split; [ reflexivity | ].
      cbn [with_extra_args].
      rewrite (lookup_none_among _ _ Hm_exn), Hnox in Hk_exn.
      cbn [orb length] in Hk_exn.
      unfold count_is in Hk_exn. rewrite Hnse in Hk_exn.
      apply Nat.eqb_eq in Hk_exn.
      destruct nse as [ | n1 [ | n2 nse]]; try discriminate Hk_exn.
      destruct (Hexn_ce n1 [] Hnse) as (-> & _).
      exact Hway_exn.
    - (* A handler in scope: an exception handler, which keeps the
         exception. *)
      rewrite Hmask1 in Hxa |- *.
      destruct (lookup (ap_exn_continuation ap) xs) as [bx | ] eqn:Hxl;
        [ | simpl in Hxa; destruct Hxa ].
      subst bx. pose proof (Hxa eq_refl) as Hhd.
      exists (filter_kept (tl mask1) extra'). split.
      + rewrite <- filter_kept_map, map_fst_rewrite.
        apply eval_simples_filter_kept. exact Hextra'.
      + right. right. left.
        exists exn, mask1, c1, c1', extra, extra'.
        split; [ reflexivity | ].
        split; [ unfold resolve; rewrite Hc1; reflexivity | ].
        split; [ unfold resolve; rewrite Hc1'; reflexivity | ].
        split; [ | split; [ exact Hcrel1 | split; [ exact Hhd | ] ] ].
        * intros i n p Hn Hp.
          destruct i as [ | [ | i]]; try discriminate Hn.
          injection Hn as <-.
          destruct (cont_params c1) as [ | p0 ps] eqn:Hcp;
            [ discriminate Hp | ].
          injection Hp as <-.
          destruct (Hexn_ce p0 ps Hce1) as (-> & _ & _).
          split; [ intros H; exact H | intros v v' H; exact H ].
        * split; [ exact Hkx | ].
          intros i p v v' Hp Hv Hv' Hu.
          destruct (cont_params c1) as [ | p0 ps] eqn:Hcp;
            [ destruct i; discriminate Hp | ].
          cbn [nth_error] in Hp.
          destruct (Hexn_ce p0 ps Hce1) as (_ & Hnoted_x & Haliases_x).
          destruct (eval_simples_nth _ _ _ _ _ Hextra Hv) as (s & Hs & Hsv).
          assert (Hal : s_alias sol (Node_variable p) (node_of_simple s)
                        = true).
          { apply (alias_input _ _ _ Hc). apply Haliases_x.
            exact (nth_error_combine _ _ i _ _ Hp Hs). }
          destruct (rewrite_simples_eval en vars vars' Henv _ extra Hnoted_x
                      Hextra)
            as (extra2 & Hextra2 & _ & Hrels_x).
          rewrite Hextra' in Hextra2. injection Hextra2 as <-.
          apply (rel_alias G sol keep called (exact = true) True is_local Hc
                   v _ _ v' Hal).
          apply (Hrels_x i s v v' Hs Hv Hv').
          destruct (is_var_used_cases p _ Hu) as [Hhu | Hk].
          -- apply kept_of_hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
          -- exact (kept_of_region en vars vars' s v Henv Hsv Hk). }
  destruct Hexn_to as (extra'' & Hextra'' & Hway_exn).
  pose proof (make_apply_wrapper_free en (rebuilt_call en ap) Hff_exn)
    as [Hret_free _].
  cbn [ap_continuation ap_exn_continuation rebuilt_call] in Hret_free.
  unfold rebuilt_call in Hconts |- *.
  match goal with
  | |- context [make_apply_wrapper first_fresh en ?a] => set (ap' := a) in *
  end.
  (* The calls of the second run, whatever they return to. *)
  assert (Hstep2 : ∀ (k1 : continuation) (conts1 : cont_env),
            resolve (ap_exn_continuation ap) conts1
            = resolve (ap_exn_continuation ap) conts' ->
            step (State (Apply (with_continuation ap' k1)) vars' conts1
                    statics' hp rcd')
            = call_function callee' Indirect vs' (ap_args_arity ap)
                (ap_return_arity ap) (resolve k1 conts1)
                (with_extra_args (resolve (ap_exn_continuation ap) conts')
                   extra'')
                statics' hp rcd').
  { intros k1 conts1 Hexn1. rewrite step_apply. unfold ap'.
    cbn [with_continuation ap_callee ap_args ap_call_kind ap_continuation
         ap_exn_continuation ap_exn_extra_args ap_args_arity ap_return_arity].
    rewrite Hcallee', Hargs_eq, Hvs'.
    unfold exn_extra_simples.
    cbn [with_continuation ap_exn_extra_args]. rewrite Hextra''.
    rewrite Hexn1. unfold rewrite_call_kind. rewrite Hkind. reflexivity. }
  (* What the engine needs. *)
  assert (Hrt_pa : cont_pa_ok statics (resolve (ap_continuation ap) conts)).
  { unfold resolve.
    destruct (lookup (ap_continuation ap) conts) as [c1 | ] eqn:Hl.
    - exact (Hconts_pa _ c1 (lookup_in _ _ _ Hl)).
    - apply Outer_pa. }
  pose proof (simple_pa_ok _ _ _ _ Henv_pa Hcallee) as Hpa_c.
  pose proof (simples_pa_ok _ _ _ _ Henv_pa Hvs) as Hvs_pa.
  pose proof (result_flow_of_ok _ Hflow') as Hrf.
  pose proof (proj2 (answered_engine (resolve (ap_continuation ap) conts)))
    as Hcalled.
  destruct (Hconts _ Hret_free)
    as [(Hm_ret & ns & Hns & Hway_ret)
       | [(rmask & c0 & c0' & Hm_ret & Hc0 & Hc0' & Hce0 & Hcrel0 & Hrn0)
         | (Kx & HKx & mx & nsx & Hmx & Lmx & Hnsx & Hrrx)]].
  3: {
    (* The return continuation of code that changes: the call returns
       there, directly or through the wrapper, with its mask. *)
    pose proof (Hcount_rets _ _ HKx eq_refl) as Hcnt.
    pose proof (proj2 (Hnr_ap _ _ HKx) eq_refl) as HK.
    unfold count_is in Hcnt. rewrite Hnsx in Hcnt. apply Nat.eqb_eq in Hcnt.
    pose proof (Hret_ce _ Hnsx) as Hrn. subst returns.
    assert (Hret_r : ∀ rt' : cont_value,
              (∀ ns' : list node,
                 length ns' = length nsx ->
                 (∀ (i : nat) (n n' : node),
                    nth_error nsx i = Some n -> nth_error ns' i = Some n' ->
                    transports n' n) ->
                 ret_rel cont_rel over_rel (repeat true (length Kx)) Kx ns'
                   (resolve (ap_continuation ap) conts) rt') ->
              ret_ok cont_rel over_rel nsx (ap_return_arity ap)
                (resolve (ap_continuation ap) conts) rt').
    { intros rt' Hr. split.
      - right. rewrite HK. intros ns' Hlen' Htr. apply Hr; [ | exact Htr ].
        rewrite Hlen', Hcnt, HK. reflexivity.
      - intros H1. left. intros E. rewrite E in Hcnt.
        rewrite H1 in Hcnt. discriminate Hcnt. }
    unfold make_apply_wrapper.
    change (ap_continuation ap') with (ap_continuation ap).
    assert (Hdirect : forallb (fun keep => keep) mx = true ->
              answered related exits_related measure
                (State (Apply ap) vars conts statics hp rcd)
                (State (Apply ap') vars' conts' statics' hp rcd')).
    { intros Hall. apply answered_of_result. rewrite Hstep1.
      change (Apply ap')
        with (Apply (with_continuation ap' (ap_continuation ap))).
      rewrite (Hstep2 (ap_continuation ap) conts' eq_refl).
      apply (Hcalled N callee callee' vs vs' _ _ _ _ _ ws nsx exn statics
               statics' hp rcd rcd' Hnex Hunt Hhu_callee Hne_a Hlen_wa
               Hrel_c Hpa_c Ha_rel Hvs_pa Hreads);
        [ | exact Hrt_pa | exact Hway_exn | exact Hst | exact Hcodes
          | exact Hsflow | exact Hrf ].
      apply Hret_r. intros ns' Hlen' Htr.
      exact (ret_rel_all_true _ _ mx Kx nsx ns' _ _ Hrrx Hall Lmx Hlen'
               Htr). }
    destruct Hmx as [Hmx | [Hmx Hall]].
    2: { rewrite Hmx. exact (Hdirect Hall). }
    rewrite Hmx.
    destruct (forallb (fun keep => keep) mx) eqn:Hall.
    + exact (Hdirect eq_refl).
    + eapply answered_of_result_after.
      * rewrite step_let_cont_non_recursive. reflexivity.
      * intros k vs0 hp0 rcd0. rewrite Hstep1.
        apply call_function_not_exit. intros Hret1 v.
        apply (ret_rel_no_exit _ _ _ _ _ Hrrx [v] statics hp rcd).
        -- cbn [length]. rewrite Hcnt, Hret1. reflexivity.
        -- left. exact (forallb_false_nth mx Hall).
      * rewrite Hstep1, (Hstep2 first_fresh).
        2: { unfold resolve. cbn [lookup].
             apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity. }
        apply (Hcalled N callee callee' vs vs' _ _ _ _ _ ws nsx
                 exn statics statics' hp rcd rcd' Hnex Hunt Hhu_callee Hne_a
                 Hlen_wa Hrel_c Hpa_c Ha_rel Hvs_pa Hreads);
          [ | exact Hrt_pa | exact Hway_exn | exact Hst | exact Hcodes
            | exact Hsflow | exact Hrf ].
        apply Hret_r. intros ns' Hlen' Htr.
        match goal with
        | |- ret_rel _ _ _ _ _ _ ?w =>
          replace w with (wrapper (ap_continuation ap) mx Kx vars' conts')
        end.
        -- apply (ret_rel_plain_wrapper _ _ mx Kx ns' nsx);
             [ exact Hrrx | exact Lmx | exact (eq_sym Hlen') | | exact Hall
             | ].
           ++ rewrite Hlen', Hcnt, HK. reflexivity.
           ++ intros i n n' Hn Hn'. exact (Htr i n' n Hn' Hn).
        -- unfold resolve, wrapper. cbn [lookup].
           rewrite Pos.eqb_refl, <- HK. reflexivity. }
  - (* Not a handler of the expression: the call returns there. *)
    pose proof Hk_ret as Hcount_ret.
    rewrite (lookup_none_among _ _ Hm_ret) in Hcount_ret.
    cbn [orb] in Hcount_ret.
    unfold count_is in Hcount_ret. rewrite Hns in Hcount_ret.
    apply Nat.eqb_eq in Hcount_ret.
    pose proof (Hret_ce ns Hns) as Hrn. subst returns.
    apply answered_of_result. rewrite Hstep1.
    unfold make_apply_wrapper.
    change (ap_continuation ap') with (ap_continuation ap).
    rewrite Hm_ret.
    change (Apply ap')
      with (Apply (with_continuation ap' (ap_continuation ap))).
    rewrite (Hstep2 (ap_continuation ap) conts' eq_refl).
    apply (Hcalled N callee callee' vs vs' _ _ _ _ _ ws ns exn statics
             statics' hp rcd rcd' Hnex Hunt Hhu_callee Hne_a Hlen_wa
               Hrel_c
             Hpa_c Ha_rel Hvs_pa Hreads);
      [ | exact Hrt_pa | exact Hway_exn | exact Hst | exact Hcodes
        | exact Hsflow | exact Hrf ].
    split.
    + left. intros ns' Hlen' Htr.
      apply (way_out_transport _ _ ns); [ exact Hway_ret | | exact Htr ].
      rewrite Hlen', Hcount_ret. reflexivity.
    + intros H1. left. intros E. rewrite E in Hcount_ret.
      rewrite H1 in Hcount_ret. discriminate Hcount_ret.
  - (* A handler of the expression: directly or through the wrapper. *)
    pose proof (Hret_ce _ Hce0) as Hrn. subst returns.
    assert (Hres0 : resolve (ap_continuation ap) conts = c0)
      by (unfold resolve; rewrite Hc0; reflexivity).
    assert (Hkc0 : cont_kinds c0 = unarize (ap_return_arity ap)).
    { destruct (Hagree _ _ Hm_ret
                  Hrn0)
        as (c2 & Hc2 & Hks2).
      rewrite Hc0 in Hc2. injection Hc2 as <-.
      rewrite Hks2 in Hks. apply kinds_eqb_eq in Hks. exact Hks. }
    destruct (cont_rel_closure _ _ _ Hcrel0)
      as (h0 & hvars0 & hconts0 & hrec0 & Hc0eq).
    assert (Hret_h : ∀ rt' : cont_value,
              (rt' = c0' ∧ forallb (fun keep => keep) rmask = true)
              ∨ (∃ (k : continuation) (wvars : env) (wconts : cont_env),
                   rt' = wrapper k rmask (cont_kinds c0) wvars wconts
                   ∧ lookup k wconts = Some c0') ->
              ret_ok cont_rel over_rel (cont_params c0) (ap_return_arity ap)
                (resolve (ap_continuation ap) conts) rt').
    { intros rt' Hrt'. rewrite Hres0. split.
      - left. intros ns' _ Htr. right. left. exists rmask, c0'.
        split; [ | split; [ exact Hcrel0 | exact Hrt' ] ].
        intros i n p Hn Hp. exact (Htr i p n Hp Hn).
      - intros _.
        destruct (cont_params c0) as [ | p0 ps] eqn:Hcp;
          [ right | left; discriminate ].
        intros v st hp0 rcd0. rewrite Hc0eq.
        rewrite Hc0eq in Hcp. destruct h0 as [params md body].
        cbn [cont_params handler_params] in Hcp.
        destruct params as [ | x params]; [ | discriminate Hcp ].
        cbn [apply_cont_value]. destruct (kinds_eqb _ _); reflexivity. }
    unfold make_apply_wrapper.
    change (ap_continuation ap') with (ap_continuation ap).
    rewrite Hm_ret.
    destruct (forallb (fun keep => keep) rmask) eqn:Hall.
    + apply answered_of_result. rewrite Hstep1.
      change (Apply ap')
        with (Apply (with_continuation ap' (ap_continuation ap))).
      rewrite (Hstep2 (ap_continuation ap) conts' eq_refl).
      apply (Hcalled N callee callee' vs vs' _ _ _ _ _ ws (cont_params c0)
               exn statics statics' hp rcd rcd' Hnex Hunt Hhu_callee Hne_a
               Hlen_wa Hrel_c Hpa_c Ha_rel Hvs_pa Hreads);
        [ | exact Hrt_pa | exact Hway_exn | exact Hst | exact Hcodes
          | exact Hsflow | exact Hrf ].
      apply Hret_h. left. split; [ | reflexivity ].
      unfold resolve. rewrite Hc0'. reflexivity.
    + eapply answered_of_result_after.
      * rewrite step_let_cont_non_recursive. reflexivity.
      * intros k vs0 hp0 rcd0. rewrite Hstep1, Hres0, Hc0eq.
        apply call_function_closure_not_exit.
      * rewrite Hstep1, (Hstep2 first_fresh).
        2: { unfold resolve. cbn [lookup].
             apply Pos.eqb_neq in Hff_exn. rewrite Hff_exn. reflexivity. }
        apply (Hcalled N callee callee' vs vs' _ _ _ _ _ ws (cont_params c0)
                 exn statics statics' hp rcd rcd' Hnex Hunt Hhu_callee Hne_a
                 Hlen_wa Hrel_c Hpa_c Ha_rel Hvs_pa Hreads);
          [ | exact Hrt_pa | exact Hway_exn | exact Hst | exact Hcodes
            | exact Hsflow | exact Hrf ].
        apply Hret_h. right.
        exists (ap_continuation ap), vars', conts'.
        split; [ | exact Hc0' ].
        unfold resolve, wrapper. cbn [lookup]. rewrite Pos.eqb_refl, Hkc0.
        reflexivity.
Qed.

(* A call: direct or indirect. *)
Lemma apply_answered (ap : apply_expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (rcd rcd' : read_record)
    (e' : expr) :
  related (State (Apply ap) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd') ->
  answered related exits_related measure
    (State (Apply ap) vars conts statics hp rcd)
    (State e' vars' conts' statics' hp' rcd').
Proof.
  destruct (ap_call_kind ap) as [id | ] eqn:Hkind.
  - exact (direct_answered ap id vars vars' conts conts' statics statics' hp hp'
    rcd rcd'
             e' Hkind).
  - exact (indirect_answered ap vars vars' conts conts' statics statics' hp hp'
    rcd rcd'
             e' Hkind).
Qed.

Lemma related_answered (s t : state) :
  related s t -> answered related exits_related measure s t.
Proof.
  rewrite (state_eta s), (state_eta t).
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros Hrel.
  - exact (let_answered _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (let_static_answered _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (let_cont_non_recursive_answered _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (let_cont_recursive_answered _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (apply_answered _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (apply_cont_answered _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (switch_answered _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - unfold answered. rewrite step_invalid. exact I.
Qed.

(* The unit refines its rebuild, given what the traversal makes of it and
   what the unit asks of it. *)
Lemma unit_refines (e : expr) :
  edges None (unit_conts u) e ->
  (∀ k : continuation, k ∈ free_continuations_of_expr e -> unit_cont k) ->
  expr_ok None [] FlambdaReaperRebuild.empty_env (unit_conts u) [] e ->
  refines_observably (State e [] [] empty_static_env empty_heap no_reads)
    (State (rebuild FlambdaReaperRebuild.empty_env e) [] [] empty_static_env
       empty_heap no_reads)
  ∧ (exact = true ->
     refines (State e [] [] empty_static_env empty_heap no_reads)
       (State (rebuild FlambdaReaperRebuild.empty_env e) [] []
          empty_static_env empty_heap no_reads)).
Proof.
  intros He Hfree Hok.
  enough (Hrel : related (State e [] [] empty_static_env empty_heap no_reads)
                   (State (rebuild FlambdaReaperRebuild.empty_env e) [] []
                      empty_static_env empty_heap no_reads)).
  { split.
    - exact (shrinking_refines_observably related exits_related measure
               related_answered _ _ exits_related_observed Hrel).
    - intros Hx.
      exact (shrinking_refines related exits_related measure related_answered
               _ _ (fun vs vs' => exits_related_eq vs vs' Hx) Hrel). }
  split; [ exact (initial_flow_invariant is_local G sol noted deps cds keep
                    codes symbols u False no_loc e He) | ].
  exists None, FlambdaReaperRebuild.empty_env, (unit_conts u), [].
  simpl_state.
  split; [ exact He | split; [ reflexivity | ] ].
  split; [ intros x v Hx; discriminate Hx | ].
  split.
  - intros k Hk.
    pose proof (Hfree k (proj1 free_rebuild e
                           (proj1 (proj2 (proj2 (proj2 Hok)))) _ k Hk))
      as Hunit_k.
    left. split; [ reflexivity | ].
    destruct (unit_conts_lookup _ Hunit_k) as (p & Hp & _).
    exists (unit_cont_nodes u k).
    split; [ unfold unit_cont_nodes; rewrite Hp; reflexivity | ].
    left. exists k.
    split; [ reflexivity | split; [ reflexivity | split; [ exact Hunit_k | ] ]
      ].
    split; [ reflexivity | ].
    intros i n p' Hn Hp'. rewrite Hn in Hp'. injection Hp' as <-.
    split; [ | intros v v' H; exact H ].
    unfold unit_cont_nodes in Hn. rewrite Hp in Hn.
    destruct i as [ | [ | i]]; simpl in Hn; try discriminate Hn.
    injection Hn as <-.
    apply (au_hu _ _ _ Hc). apply (any_usage_input _ _ _ Hc).
    apply (Hunit k p). unfold unit_cont_nodes. rewrite Hp. left. reflexivity.
  - split.
    + split; [ intros s _; reflexivity | ].
      split; [ intros s tag fields Hl; discriminate Hl | ].
      split; [ intros s allocated Hl; discriminate Hl | ].
      intros id. left. reflexivity.
    + split; [ reflexivity | split; [ exact Hok | left; reflexivity ] ].
Qed.

End Run.

(* ======================================================================== *)
(* The facts that the traversal makes of the [Switch]es                     *)
(* ======================================================================== *)

Lemma flat_map_nil {A B : Type} (l : list A) :
  flat_map (fun _ : A => @nil B) l = [].
Proof. induction l as [ | a l IH]; [ reflexivity | exact IH ]. Qed.

(* A step of the traversal only adds to what it is given, and makes [P]
   of what follows. *)
Definition grows (step : acc -> acc) (P : acc -> Prop) : Prop :=
  (∀ a : acc, acc_le a (step a)) ∧ ∀ a a' : acc, acc_le (step a) a' -> P a'.

Lemma makes_grows (ids : list code_id) (step : acc -> acc) (P : acc -> Prop) :
  makes ids step P -> grows step P.
Proof.
  intros H. split; [ intros a; exact (proj1 (proj1 (H a))) | ].
  intros a a' Hle. exact (proj2 (H a) a' Hle).
Qed.

Lemma grows_compose (step step' : acc -> acc) (P P' : acc -> Prop) :
  grows step P -> grows step' P' ->
  grows (fun a => step' (step a)) (fun a' => P a' ∧ P' a').
Proof.
  intros [H1 H2] [H1' H2']. split.
  - intros a. exact (acc_le_trans _ _ _ (H1 a) (H1' (step a))).
  - intros a a' Hle. split; [ | exact (H2' (step a) a' Hle) ].
    apply (H2 a). exact (acc_le_trans _ _ _ (H1' (step a)) Hle).
Qed.

Lemma grows_weaken (step : acc -> acc) (P P' : acc -> Prop) :
  (∀ a : acc, P a -> P' a) -> grows step P -> grows step P'.
Proof.
  intros H [H1 H2]. split; [ exact H1 | ].
  intros a a' Hle. exact (H _ (H2 a a' Hle)).
Qed.

Lemma grows_ext (step step' : acc -> acc) (P : acc -> Prop) :
  (∀ a : acc, step' a = step a) -> grows step P -> grows step' P.
Proof.
  intros He [H1 H2]. split.
  - intros a. rewrite He. exact (H1 a).
  - intros a a' Hle. rewrite He in Hle. exact (H2 a a' Hle).
Qed.

Lemma grows_fold {A : Type} (f : acc -> A -> acc) (P : A -> acc -> Prop)
    (l : list A) :
  (∀ x : A, x ∈ l -> grows (fun a => f a x) (P x)) ->
  grows (fun a => List.fold_left f l a) (fun a' => ∀ x : A, x ∈ l -> P x a').
Proof.
  induction l as [ | x l IH]; intros H.
  - split; [ intros a; apply acc_le_refl | intros a a' _ x [] ].
  - pose proof (grows_compose _ _ _ _ (H x (or_introl eq_refl))
                  (IH (fun y Hy => H y (or_intror Hy)))) as Hc.
    simpl. eapply grows_weaken; [ | exact Hc ].
    intros a [Hx Hl] y [<- | Hy]; [ exact Hx | exact (Hl y Hy) ].
Qed.

Lemma prepare_bindings_le (l : list static_binding) (a : acc) :
  acc_le a (List.fold_left (fun a b => prepare_binding b a) l a).
Proof.
  revert a. induction l as [ | b l IH]; intros a; [ apply acc_le_refl | ].
  simpl. apply (acc_le_trans _ (prepare_binding b a)); [ | apply IH ].
  destruct b as [id c | id | s const]; simpl;
    [ | apply acc_le_refl | apply acc_le_refl ].
  destruct (prepare_code_spec id c a) as (cd & _ & Hle & _). exact Hle.
Qed.

(* Once the bindings are prepared, the code they bind has a code dep. *)
Lemma prepare_bindings_deps (l : list static_binding) (a : acc)
    (id : code_id) (c : code) :
  Bind_code id c ∈ l ->
  ∃ cd : code_dep,
    (id, cd) ∈ acc_code_deps
                 (List.fold_left (fun a b => prepare_binding b a) l a).
Proof.
  revert a. induction l as [ | b l IH]; intros a Hin; [ destruct Hin | ].
  simpl. destruct Hin as [-> | Hin]; [ | exact (IH _ Hin) ].
  simpl. destruct (prepare_code_spec id c a) as (cd & Hcds & _).
  exists cd. apply (proj2 (proj2 (proj2 (prepare_bindings_le l _)))).
  rewrite Hcds. left. reflexivity.
Qed.

(* The scrutinee of every [Switch] of a term of the check has any usage
   at toplevel ([cond_any_usage]), and in code is used by the code
   id. *)
Lemma traverse_scrutinees (ex : bool) (codes : list (code_id * code)) :
  (∀ e : expr,
     checked ex codes e = true ->
     ∀ env : traverse_env,
       (∀ w : value_slot,
          w ∈ projected_value_slots e -> should_keep_value_slot env w = true) ->
       grows (traverse env e)
         (fun a' => scrutinees_used (acc_graph a') (env_current_code_id env) e))
  ∧ (∀ h : continuation_handler,
       handler_checked ex codes h = true ->
       ∀ env : traverse_env,
         (∀ w : value_slot,
            w ∈ handler_projected_value_slots h ->
            should_keep_value_slot env w = true) ->
         grows (traverse_handler env h)
           (fun a' =>
              handler_scrutinees_used (acc_graph a') (env_current_code_id env)
                h)).
Proof.
  apply expr_handler_ind.
  - (* [Let] *)
    intros x kind d body IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hd Hbody].
    apply named_checked_no_code in Hd.
    assert (Hmk : makes [] (traverse_named env x d) (fun _ => True)).
    { destruct (named_in_flow_fragment false d) eqn:Hd'.
      - eapply makes_weaken; [ intros; exact I | ].
        exact (traverse_named_makes env True false x d Hd' (fun _ => I)
                 (fun w Hw => Hkeep w (in_or_app _ _ _ (or_introl Hw)))).
      - destruct d as [ | p args | | | | | ]; try discriminate Hd;
          try discriminate Hd'.
        eapply makes_weaken; [ intros; exact I | ].
        exact (traverse_prim_makes_any env x p args). }
    pose proof (grows_compose _ _ _ _
                  (makes_grows _ _ _ Hmk)
                  (IH Hbody env
                     (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw)))))
      as Hboth.
    eapply grows_weaken; [ | exact Hboth ]. intros a' [_ H]. exact H.
  - (* [Let_static] *)
    intros bindings body Hcodes IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hbindings Hbody].
    rewrite forallb_forall in Hbindings.
    assert (Hprep :
      grows (fun a => List.fold_left (fun a b => prepare_binding b a) bindings
                        a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros b _. split; [ | intros; exact I ].
      intros a. destruct b as [id c | id | s const]; simpl;
        [ | apply acc_le_refl | apply acc_le_refl ].
      destruct (prepare_code_spec id c a) as (cd & _ & Hle & _). exact Hle. }
    assert (Hstat :
      grows (fun a =>
               List.fold_left (fun a b => traverse_static_binding env b a)
                 bindings a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros b Hin. specialize (Hbindings b Hin).
      destruct (binding_checked_cases _ _ _ Hbindings)
        as [Hnc | (id & c & -> & _ & _)];
        [ | split; [ intros a; apply acc_le_refl | intros; exact I ] ].
      apply (makes_grows []).
      destruct b as [ | id | s [ | slot syms decls vs | t fields | contents
                                 | bytes | elements ]];
        simpl in Hnc; try discriminate Hnc.
      + apply makes_steady; [ intros a; apply steady_refl | intros; exact I ].
      + eapply makes_weaken;
          [ | apply (static_set_makes env (map snd syms)) ];
          [ intros; exact I | ].
        apply andb_true_iff in Hnc as [Hbound _].
        rewrite Hbound. simpl. apply forallb_forall. intros [g sym] Hgs.
        simpl. apply among_In. apply in_map_iff. exists (g, sym). auto.
      + eapply makes_weaken;
          [ | exact (block_deps_makes (Node_symbol s) fields) ].
        intros; exact I.
      + apply makes_steady;
          [ intros a; apply add_facts_steady | intros; exact I ].
      + apply makes_steady;
          [ intros a; apply add_facts_steady | intros; exact I ].
      + destruct elements as [ | el els ];
          [ apply makes_steady;
            [ intros a; apply add_facts_steady | intros; exact I ] | ].
        eapply makes_weaken;
          [ | exact (block_deps_makes (Node_symbol s) (el :: els)) ].
        intros; exact I. }
    assert (Hcode_env :
      ∀ (id : code_id) (params : list variable) (mc : variable)
        (ret exn : continuation) (md : code_metadata) (cbody : expr)
        (cd : code_dep),
        Bind_code id (Code params mc ret exn md cbody) ∈ bindings ->
        grows (traverse (code_env env id ret exn cd) cbody)
          (fun a' => scrutinees_used (acc_graph a') (Some id) cbody)).
    { intros id params mc ret exn md cbody cd Hin.
      destruct (binding_checked_cases _ _ _ (Hbindings _ Hin))
        as [Hnc | (id' & c' & Heq & He & Hnc)]; [ discriminate Hnc | ].
      injection Heq as <- <-.
      assert (Hcb : checked ex codes cbody = true) by (rewrite He; exact Hnc).
      pose proof (Hcodes id _ Hin Hcb) as IHc.
      refine (IHc (code_env env id ret exn cd) _). intros w Hw. apply Hkeep.
      apply in_or_app. left.
      apply in_flat_map.
      exists (Bind_code id (Code params mc ret exn md cbody)).
      split; [ exact Hin | exact Hw ]. }
    assert (Hbind1 : ∀ (b : static_binding) (a : acc),
              b ∈ bindings -> acc_le a (traverse_binding env b a)).
    { intros b a Hin.
      destruct b as [id c | id | s const]; simpl;
        [ | apply acc_le_refl | apply acc_le_refl ].
      destruct c as [params mc ret exn md cbody]. simpl.
      destruct (lookup id (acc_code_deps a)) as [cd | ];
        [ | apply acc_le_refl ].
      match goal with
      | |- acc_le a (traverse _ _ (add_facts ?l a)) =>
        apply (acc_le_trans _ (add_facts l a));
          [ exact (proj1 (add_facts_steady l a)) | ]
      end.
      exact (proj1 (Hcode_env id params mc ret exn md cbody cd Hin) _). }
    assert (Hfold_le : ∀ (l : list static_binding) (a : acc),
              (∀ b : static_binding, b ∈ l -> b ∈ bindings) ->
              acc_le a
                (List.fold_left (fun a b => traverse_binding env b a) l a)).
    { induction l as [ | b l IHl]; intros a Hsub; [ apply acc_le_refl | ].
      simpl. apply (acc_le_trans _ (traverse_binding env b a));
        [ exact (Hbind1 b a (Hsub b (or_introl eq_refl))) | ].
      apply IHl. intros b' Hb'. exact (Hsub b' (or_intror Hb')). }
    assert (Hcode_scr :
      ∀ (l : list static_binding) (a a' : acc),
        (∀ b : static_binding, b ∈ l -> b ∈ bindings) ->
        (∀ (id : code_id) (c : code),
           Bind_code id c ∈ l -> ∃ cd : code_dep, (id, cd) ∈ acc_code_deps a) ->
        acc_le (List.fold_left (fun a b => traverse_binding env b a) l a) a' ->
        every (binding_scrutinees_used (acc_graph a')) l).
    { induction l as [ | b l IHl]; intros a a' Hsub Hdeps Hle; [ exact I | ].
      simpl in Hle.
      assert (Hle1 : acc_le (traverse_binding env b a) a').
      { eapply acc_le_trans; [ | exact Hle ].
        apply Hfold_le. intros b' Hb'. exact (Hsub b' (or_intror Hb')). }
      split.
      - destruct b as [id c | id | s const]; [ | exact I | exact I ].
        destruct (Hdeps id c (or_introl eq_refl)) as (cd0 & Hcd0).
        destruct c as [params mc ret exn md cbody].
        cbn [traverse_binding traverse_code] in Hle1.
        destruct (lookup id (acc_code_deps a)) as [cd | ] eqn:Hl.
        + exact (proj2 (Hcode_env id params mc ret exn md cbody cd
                          (Hsub _ (or_introl eq_refl))) _ a' Hle1).
        + exfalso. apply (lookup_none_not_in id _ Hl).
          exact (in_map fst _ (id, cd0) Hcd0).
      - apply (IHl (traverse_binding env b a) a').
        + intros b' Hb'. exact (Hsub b' (or_intror Hb')).
        + intros id c Hin. destruct (Hdeps id c (or_intror Hin)) as (cd & Hcd).
          exists cd.
          exact (proj2 (proj2 (proj2 (Hbind1 b a (Hsub b (or_introl eq_refl)))))
                   _ Hcd).
        + exact Hle. }
    assert (Hbind :
      grows (fun a => List.fold_left (fun a b => traverse_binding env b a)
                        bindings a)
        (fun _ => True)).
    { split; [ intros a; exact (Hfold_le bindings a (fun b Hb => Hb)) | ].
      intros; exact I. }
    pose proof (grows_compose _ _ _ _
                  (grows_compose _ _ _ _ (grows_compose _ _ _ _ Hprep Hstat)
                     Hbind)
                  (IH Hbody env
                     (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw)))))
      as Hall.
    split; [ exact (proj1 Hall) | ].
    intros a a' Hle. split; [ | exact (proj2 (proj2 Hall a a' Hle)) ].
    set (a1 := List.fold_left (fun a b => prepare_binding b a) bindings a).
    set (a2 := List.fold_left (fun a b => traverse_static_binding env b a)
                 bindings a1).
    apply (Hcode_scr bindings a2 a' (fun b Hb => Hb)).
    + intros id c Hin. destruct (prepare_bindings_deps bindings a id c Hin)
        as (cd & Hcd).
      exists cd. exact (proj2 (proj2 (proj2 (proj1 Hstat a1))) _ Hcd).
    + eapply acc_le_trans; [ | exact Hle ].
      exact (proj1 (IH Hbody env
                      (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw))))
               _).
  - (* [Let_cont_non_recursive] *)
    intros k h body IHh IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hh Hbody].
    pose proof (grows_compose _ _ _ _
                  (grows_compose _ _ _ _
                     (IHh Hh env
                        (fun w Hw => Hkeep w (in_or_app _ _ _ (or_introl Hw))))
                     (makes_grows _ _ _
                        (makes_steady _ (fun _ => True)
                           (exn_handler_deps_steady env h)
                           (fun _ _ _ => I))))
                  (IH Hbody (add_cont env k (handler_params h))
                     (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw)))))
      as Hboth.
    eapply grows_weaken; [ | exact Hboth ].
    intros a' [[Hh' _] Hb]. split; [ exact Hh' | exact Hb ].
  - (* [Let_cont_recursive] *)
    intros handlers body IHh IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    set (env' := add_rec_conts env handlers).
    assert (Hcode' : env_current_code_id env' = env_current_code_id env)
      by exact (proj1 (add_rec_conts_fields env handlers)).
    assert (Hkeep' : ∀ w : value_slot,
              w ∈ projected_value_slots (Let_cont_recursive handlers body) ->
              should_keep_value_slot env' w = true).
    { intros w Hw. unfold should_keep_value_slot, env'.
      rewrite (proj2 (add_rec_conts_fields env handlers)). exact (Hkeep w Hw). }
    assert (Hfold :
      grows
        (fun a =>
           List.fold_left (fun a '(_, h) => traverse_handler env' h a)
             handlers a)
        (fun a' =>
           ∀ kh, kh ∈ handlers ->
             let '(_, h) := kh in
             handler_scrutinees_used (acc_graph a') (env_current_code_id env')
               h)).
    { apply (grows_fold _
               (fun kh a' =>
                  let '(_, h) := kh in
                  handler_scrutinees_used (acc_graph a')
                    (env_current_code_id env') h)).
      intros [k h] Hin.
      apply (IHh k h Hin (Hhandlers _ Hin) env').
      intros w Hw. apply Hkeep'. apply in_or_app. left.
      apply in_flat_map. exists (k, h). auto. }
    pose proof (grows_compose _ _ _ _ Hfold
                  (IH Hbody env'
                     (fun w Hw => Hkeep' w (in_or_app _ _ _ (or_intror Hw)))))
      as Hboth.
    eapply grows_ext; [ | eapply grows_weaken; [ | exact Hboth ] ];
      [ intros a; reflexivity | ].
    intros a' [Hall Hb]. rewrite Hcode' in Hb.
    split; [ | exact Hb ].
    apply (every_in _ _ _). intros [k h] Hin.
    pose proof (Hall (k, h) Hin) as Hkh. simpl in Hkh. rewrite Hcode' in Hkh.
    exact Hkh.
  - (* [Apply] *)
    intros ap _ env _.
    eapply grows_weaken;
      [ | apply (makes_grows []); apply (traverse_apply_makes env []) ].
    + intros; exact I.
    + unfold apply_in_flow_fragment. destruct (ap_call_kind ap); reflexivity.
  - (* [Apply_cont] *)
    intros ac _ env _.
    eapply grows_weaken;
      [ | apply (makes_grows []); apply apply_cont_deps_makes ].
    intros; exact I.
  - (* [Switch] *)
    intros scrutinee arms _ env _.
    assert (Hcond :
      grows (cond_any_usage env scrutinee)
        (fun a' =>
           scrutinees_used (acc_graph a') (env_current_code_id env)
             (Switch scrutinee arms))).
    { apply (makes_grows []). apply makes_steady;
        [ apply cond_any_usage_steady | ].
      intros a a' Hle. unfold cond_any_usage in Hle. simpl.
      destruct (env_current_code_id env) as [id | ].
      - apply facts_use.
        exact (add_facts_later _ _ _ _ Hle (or_introl eq_refl)).
      - apply facts_any_usage.
        exact (add_facts_later _ _ _ _ Hle (or_introl eq_refl)). }
    assert (Hfold :
      grows
        (fun a =>
           List.fold_left (fun a '(_, ac) => apply_cont_deps env ac a)
             (Keyed.compact Z.eqb arms) a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros [d ac] _.
      eapply grows_weaken;
        [ | apply (makes_grows []); apply apply_cont_deps_makes ].
      intros; exact I. }
    pose proof (grows_compose _ _ _ _ Hcond Hfold) as Hboth.
    eapply grows_weaken; [ | exact Hboth ]. intros a' [H _]. exact H.
  - (* [Invalid] *)
    intros _ env _.
    split; [ intros a; apply acc_le_refl | intros; exact I ].
  - (* A handler *)
    intros params metadata handler IH Hcf env Hkeep.
    exact (IH Hcf env Hkeep).
Qed.

Lemma traverse_unit_scrutinees (ex : bool) (codes : list (code_id * code))
    (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr) :
  checked ex codes e = true ->
  scrutinees_used (graph_of_facts (traverse_unit flag u e)) None e.
Proof.
  intros Hcf. rewrite traverse_unit_eq.
  destruct (proj1 (traverse_scrutinees ex codes) e Hcf (initial_env flag u e)
              (fun w Hw => proj2 (among_In _ _) Hw))
    as [_ Hscr].
  set (a4 := traverse (initial_env flag u e) e initial_acc) in *.
  set (a5 := add_fact (Fact_any_usage (Node_made 1)) a4).
  destruct (resolve_delayed_deps_spec (static_names symbols_of_binding e) a5)
    as (S6 & _).
  exact (Hscr initial_acc _
           (proj1 (steady_trans _ _ _ (add_facts_steady _ a4) S6))).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit without closures meets [no_code_no_slots]. *)
Theorem closure_free_no_code_no_slots :
  closure_free_no_code_no_slots_statement.
Proof.
  enough (H : (∀ e : expr, closure_free e = true -> no_code_no_slots e = true)
              ∧ (∀ h : continuation_handler,
                   handler_closure_free h = true ->
                   handler_no_code_no_slots h = true))
    by exact (proj1 H).
  apply expr_handler_ind.
  - intros x kind d body IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hd Hb]. rewrite (IH Hb), andb_true_r.
    destruct d; try discriminate Hd; reflexivity.
  - intros bindings body _ IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [ | | s [ | | t fields | | | ]]; try discriminate Hbs.
    reflexivity.
  - intros k h body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - intros ac H. exact H.
  - intros scrutinee arms H. exact H.
  - intros H. exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* The [Apply_cont]s of a unit that gives the continuations of the unit
   one value, of continuations that are the unit's or handlers in scope,
   give one value to the first; and its calls that do not return, or
   raise, to handlers in scope give one value to a continuation of the
   unit. *)
Lemma exits_bound_of_exits (u : unit_interface) :
  (∀ e : expr,
     exits_one_value u e = true ->
     ∀ (bound : list continuation) (ce : finite_map (list node)),
       (∀ k : continuation,
          k ∈ free_continuations_of_expr e ->
          unit_cont u k ∨ among k bound = true) ->
       (∀ k : continuation,
          unit_cont u k -> among k bound = false ->
          ∃ p : node, lookup k ce = Some [p]) ->
       exits_bound bound ce e = true)
  ∧ (∀ h : continuation_handler,
       handler_exits_one_value u h = true ->
       ∀ (bound : list continuation) (ce : finite_map (list node)),
         (∀ k : continuation,
            k ∈ free_continuations_of_handler h ->
            unit_cont u k ∨ among k bound = true) ->
         (∀ k : continuation,
            unit_cont u k -> among k bound = false ->
            ∃ p : node, lookup k ce = Some [p]) ->
         handler_exits_bound bound ce h = true).
Proof.
  assert (Hcount : ∀ (k : continuation) (bound : list continuation)
                     (ce : finite_map (list node)),
            unit_cont u k ->
            (∀ k : continuation,
               unit_cont u k -> among k bound = false ->
               ∃ p : node, lookup k ce = Some [p]) ->
            among k bound = false -> count_is ce k 1 = true).
  { intros k bound ce Hu Hce Hb. destruct (Hce k Hu Hb) as (p & Hp).
    unfold count_is. rewrite Hp. reflexivity. }
  assert (Hone : ∀ k : continuation,
            unit_cont u k -> among k [unit_return u; unit_exn u] = true).
  { intros k Hu. apply among_In.
    destruct Hu as [-> | ->]; [ left | right; left ]; reflexivity. }
  assert (Hac : ∀ (ac : apply_cont) (bound : list continuation)
                  (ce : finite_map (list node)),
            apply_cont_exits_one_value u ac = true ->
            unit_cont u (ac_continuation ac)
            ∨ among (ac_continuation ac) bound = true ->
            (∀ k : continuation,
               unit_cont u k -> among k bound = false ->
               ∃ p : node, lookup k ce = Some [p]) ->
            exits_ac bound ce ac = true).
  { intros ac bound ce He Hk Hce. unfold exits_ac.
    destruct (among (ac_continuation ac) bound) eqn:Hb; [ reflexivity | ].
    destruct Hk as [Hu | Hb']; [ | discriminate Hb' ].
    unfold apply_cont_exits_one_value in He. rewrite (Hone _ Hu) in He.
    apply Nat.eqb_eq in He. rewrite He. exact (Hcount _ _ _ Hu Hce Hb). }
  apply expr_handler_ind.
  - intros x kind d body IH He bound ce Hk Hce. exact (IH He bound ce Hk Hce).
  - intros bindings body _ IH He bound ce Hk Hce. simpl in He |- *.
    apply (IH He bound ce); [ | exact Hce ]. intros k Hin. apply Hk.
    simpl. apply in_or_app. right. exact Hin.
  - intros k0 h body IHh IH He bound ce Hk Hce. simpl in He |- *.
    apply andb_true_iff in He as [Hh Hb]. apply andb_true_intro. split.
    + apply (IHh Hh bound ce); [ | exact Hce ]. intros k Hin. apply Hk.
      simpl. apply in_or_app. left. exact Hin.
    + apply (IH Hb (k0 :: bound) _).
      * intros k Hin. destruct (Pos.eqb_spec k k0) as [-> | Hne].
        -- right. unfold among. simpl. rewrite Pos.eqb_refl. reflexivity.
        -- destruct (Hk k) as [Hu | Hb'].
           ++ simpl. apply in_or_app. right. unfold without.
              apply filter_In. split; [ exact Hin | ]. unfold among. simpl.
              rewrite (proj2 (Pos.eqb_neq k k0) Hne). reflexivity.
           ++ left. exact Hu.
           ++ right. unfold among in Hb' |- *. simpl. rewrite Hb', orb_true_r.
              reflexivity.
      * intros k Hu Hb'. rewrite among_cons in Hb'.
        apply orb_false_iff in Hb' as [Hk0 Hb']. simpl. rewrite Hk0.
        exact (Hce k Hu Hb').
  - intros handlers body IHh IH He bound ce Hk Hce. simpl in He |- *.
    apply andb_true_iff in He as [Hhs Hb]. rewrite forallb_forall in Hhs.
    assert (Hk' : ∀ k : continuation,
              k ∈ flat_map (fun '(_, h) => free_continuations_of_handler h)
                     handlers
                  ++ free_continuations_of_expr body ->
              unit_cont u k ∨ among k (map fst handlers ++ bound) = true).
    { intros k Hin. rewrite among_app.
      destruct (among k (map fst handlers)) eqn:Hm;
        [ right; reflexivity | ].
      destruct (Hk k) as [Hu | Hb']; [ | left; exact Hu | right; exact Hb' ].
      simpl. unfold without. apply filter_In. split; [ exact Hin | ].
      rewrite Hm. reflexivity. }
    assert (Hce' : ∀ k : continuation,
              unit_cont u k -> among k (map fst handlers ++ bound) = false ->
              ∃ p : node, lookup k (rec_conts ce handlers) = Some [p]).
    { intros k Hu Hb'. rewrite among_app in Hb'.
      apply orb_false_iff in Hb' as [Hkh Hb']. rewrite lookup_rec_conts.
      destruct (lookup k handlers) as [h | ] eqn:Hl;
        [ | exact (Hce k Hu Hb') ].
      exfalso. apply lookup_in in Hl. apply (in_map fst) in Hl.
      apply among_In in Hl. apply not_true_iff_false in Hkh. exact (Hkh Hl). }
    apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hin.
      apply (IHh k h Hin (Hhs _ Hin) _ _); [ | exact Hce' ].
      intros k' Hk'in. apply Hk'. apply in_or_app. left. apply in_flat_map.
      exists (k, h). auto.
    + apply (IH Hb _ _); [ | exact Hce' ]. intros k Hin. apply Hk'.
      apply in_or_app. right. exact Hin.
  - intros ap He bound ce Hk Hce.
    cbn [exits_one_value] in He. cbn [exits_bound].
    unfold apply_exits_one_value in He. apply andb_true_iff in He as [Hr Hx].
    apply andb_true_intro. split.
    + destruct (among (ap_exn_continuation ap) bound) eqn:Hb;
        [ reflexivity | ].
      destruct (Hk (ap_exn_continuation ap) (or_intror (or_introl eq_refl)))
        as [Hu | Hb']; [ | congruence ].
      rewrite (Hone _ Hu) in Hx. apply Nat.eqb_eq in Hx. rewrite Hx.
      exact (Hcount _ _ _ Hu Hce Hb).
    + destruct (among (ap_continuation ap) bound) eqn:Hb; [ reflexivity | ].
      destruct (Hk (ap_continuation ap) (or_introl eq_refl)) as [Hu | Hb'];
        [ | congruence ].
      rewrite (Hone _ Hu) in Hr. apply Nat.eqb_eq in Hr. rewrite Hr.
      exact (Hcount _ _ _ Hu Hce Hb).
  - intros ac He bound ce Hk Hce. apply (Hac ac bound ce He); [ | exact Hce ].
    apply Hk. left. reflexivity.
  - intros scrutinee arms He bound ce Hk Hce. simpl in He |- *.
    rewrite forallb_forall in He |- *. intros [i ac] Hin.
    apply (Hac ac bound ce (He _ Hin)); [ | exact Hce ]. apply Hk. simpl.
    apply in_flat_map. exists (i, ac). split; [ exact Hin | left; reflexivity ].
  - intros _ bound ce _ _. reflexivity.
  - intros params metadata handler IH He bound ce Hk Hce.
    exact (IH He bound ce Hk Hce).
Qed.

Lemma exits_bound_unit (u : unit_interface) (e : expr) :
  exits_one_value u e = true ->
  (∀ k : continuation, k ∈ free_continuations_of_expr e -> unit_cont u k) ->
  exits_bound [] (unit_conts u) e = true.
Proof.
  intros He Hk. apply (proj1 (exits_bound_of_exits u) e He [] _).
  - intros k Hin. left. exact (Hk k Hin).
  - intros k Hu _. destruct (unit_conts_lookup u k Hu) as (p & Hp & _).
    exists p. exact Hp.
Qed.

(* The calls of a unit whose exception handlers are apart from its other
   continuations raise to exception handlers of non-recursive
   [Let_cont]s, and give extra arguments only to handlers in scope: in
   [e], [xs] has a handler said not to be one ([false]) only if nothing
   raises to it, and has every continuation that [e] gives extra
   arguments to. *)
Lemma raises_kept_of_well_formed :
  (∀ (e : expr) (xs : finite_map bool),
     exn_handlers_well_formed e = true ->
     (∀ k : continuation,
        lookup k xs = Some false -> raises_to k e = false) ->
     (∀ k : continuation,
        lookup k xs = None -> k ∈ free_continuations e ->
        gives_extra_args k e = false) ->
     raises_kept xs e = true)
  ∧ (∀ (h : continuation_handler) (xs : finite_map bool),
       handler_exn_handlers_well_formed h = true ->
       (∀ k : continuation,
          lookup k xs = Some false -> handler_raises_to k h = false) ->
       (∀ k : continuation,
          lookup k xs = None -> k ∈ handler_free_continuations h ->
          handler_gives_extra_args k h = false) ->
       handler_raises_kept xs h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH xs Hw H1 H2. exact (IH xs Hw H1 H2).
  - intros bindings body Hcodes IH xs Hw H1 H2. simpl in Hw |- *.
    apply andb_true_iff in Hw as [Hbs Hw].
    rewrite (IH xs Hw H1 H2), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *; try reflexivity.
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. simpl in Hbs |- *.
    apply andb_true_iff in Hbs as [Hbs Hwc].
    apply andb_true_iff in Hbs as [Hways _].
    apply (IHc [] Hwc); [ intros k Hl; discriminate Hl | ].
    intros k _ Hk. unfold ways_out_well_formed in Hways.
    rewrite forallb_forall in Hways. specialize (Hways k Hk).
    apply andb_true_iff in Hways as [_ Hg]. apply negb_true_iff in Hg.
    exact Hg.
  - intros k h body IHh IH xs Hw H1 H2. simpl in Hw |- *.
    apply andb_true_iff in Hw as [Hw Hwb].
    apply andb_true_iff in Hw as [Hk Hwh].
    rewrite (IHh xs Hwh); cbn [andb].
    + apply (IH _ Hwb).
      * intros j Hj. simpl in Hj. destruct (Pos.eqb_spec j k) as [E | Hne].
        -- rewrite E. injection Hj as Hj. rewrite Hj in Hk. simpl in Hk.
           apply negb_true_iff in Hk. exact Hk.
        -- pose proof (H1 j Hj) as H. simpl in H.
           apply orb_false_iff in H as [_ H]. exact H.
      * intros j Hj Hin. simpl in Hj.
        destruct (Pos.eqb_spec j k) as [E | Hne]; [ discriminate Hj | ].
        assert (Hin' : j ∈ free_continuations
                             (Let_cont_non_recursive k h body)).
        { simpl. apply in_or_app. right. unfold without. apply filter_In.
          split; [ exact Hin | ]. rewrite among_cons.
          apply Pos.eqb_neq in Hne. rewrite Hne. reflexivity. }
        pose proof (H2 j Hj Hin') as H. simpl in H.
        apply orb_false_iff in H as [_ H]. exact H.
    + intros j Hj. pose proof (H1 j Hj) as H. simpl in H.
      apply orb_false_iff in H as [H _]. exact H.
    + intros j Hj Hin.
      assert (Hin' : j ∈ free_continuations
                           (Let_cont_non_recursive k h body))
        by (simpl; apply in_or_app; left; exact Hin).
      pose proof (H2 j Hj Hin') as H. simpl in H.
      apply orb_false_iff in H as [H _]. exact H.
  - intros handlers body IHh IH xs Hw H1 H2. simpl in Hw |- *.
    apply andb_true_iff in Hw as [Hws Hwb]. rewrite forallb_forall in Hws.
    set (xs' := map (fun '(k, _) => (k, false)) handlers ++ xs).
    assert (HR : ∀ j : continuation,
              lookup j xs' = Some false ->
              raises_to j body = false
              ∧ ∀ (k' : continuation) (h' : continuation_handler),
                  (k', h') ∈ handlers -> handler_raises_to j h' = false).
    { intros j Hj. unfold xs' in Hj. rewrite lookup_rec_false in Hj.
      assert (Hno : ∀ l : list (continuation * continuation_handler),
                existsb (fun '(_, h) => handler_raises_to j h) l = false ->
                ∀ (k' : continuation) (h' : continuation_handler),
                  (k', h') ∈ l -> handler_raises_to j h' = false).
      { intros l Hl k' h' Hin. apply not_true_iff_false. intros E.
        apply not_true_iff_false in Hl. apply Hl. apply existsb_exists.
        exists (k', h'). split; [ exact Hin | exact E ]. }
      destruct (lookup j handlers) as [hj | ] eqn:Hl.
      - apply lookup_in in Hl. specialize (Hws _ Hl). cbv beta iota in Hws.
        repeat rewrite andb_true_iff in Hws.
        destruct Hws as [[[_ Hb] Hhs] _].
        apply negb_true_iff in Hb, Hhs.
        split; [ exact Hb | exact (Hno handlers Hhs) ].
      - pose proof (H1 j Hj) as H. simpl in H.
        apply orb_false_iff in H as [Hhs Hb].
        split; [ exact Hb | exact (Hno handlers Hhs) ]. }
    assert (HG : ∀ j : continuation,
              lookup j xs' = None ->
              (j ∈ free_continuations body ->
               gives_extra_args j body = false)
              ∧ ∀ (k' : continuation) (h' : continuation_handler),
                  (k', h') ∈ handlers ->
                  j ∈ handler_free_continuations h' ->
                  handler_gives_extra_args j h' = false).
    { intros j Hj. unfold xs' in Hj. rewrite lookup_rec_false in Hj.
      destruct (lookup j handlers) as [hj | ] eqn:Hl; [ discriminate Hj | ].
      pose proof (lookup_none_among _ _ Hl) as Hnot.
      assert (Hgo : j ∈ flat_map (fun '(_, h) => handler_free_continuations h)
                          handlers
                        ++ free_continuations body ->
                    gives_extra_args j (Let_cont_recursive handlers body)
                    = false).
      { intros Hin. apply (H2 j Hj). simpl. unfold without.
        apply filter_In. split; [ exact Hin | ].
        apply negb_true_iff. exact Hnot. }
      simpl in Hgo. split.
      - intros Hin.
        specialize (Hgo (in_or_app _ _ _ (or_intror Hin))).
        apply orb_false_iff in Hgo as [_ H]. exact H.
      - intros k' h' Hin Hin'.
        assert (Hin2 : j ∈ flat_map
                             (fun '(_, h) => handler_free_continuations h)
                             handlers)
          by (apply in_flat_map; exists (k', h'); split; assumption).
        specialize (Hgo (in_or_app _ _ _ (or_introl Hin2))).
        apply orb_false_iff in Hgo as [H _].
        apply not_true_iff_false. intros E. apply not_true_iff_false in H.
        apply H. apply existsb_exists. exists (k', h').
        split; [ exact Hin | exact E ]. }
    apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hin.
      pose proof (Hws _ Hin) as Hwk. cbv beta iota in Hwk.
      repeat rewrite andb_true_iff in Hwk. destruct Hwk as [_ Hwh].
      apply (IHh k h Hin xs' Hwh).
      * intros j Hj. exact (proj2 (HR j Hj) k h Hin).
      * intros j Hj Hin'. exact (proj2 (HG j Hj) k h Hin Hin').
    + apply (IH xs' Hwb).
      * intros j Hj. exact (proj1 (HR j Hj)).
      * intros j Hj. exact (proj1 (HG j Hj)).
  - intros ap xs Hw H1 H2. simpl.
    destruct (lookup (ap_exn_continuation ap) xs) as [[ | ] | ] eqn:Hl.
    + reflexivity.
    + exfalso. specialize (H1 _ Hl). simpl in H1.
      rewrite Pos.eqb_refl in H1. discriminate H1.
    + specialize (H2 _ Hl (or_intror (or_introl eq_refl))). simpl in H2.
      rewrite Pos.eqb_refl in H2. simpl in H2.
      apply negb_false_iff in H2. exact H2.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH xs Hw H1 H2. exact (IH xs Hw H1 H2).
Qed.

(* The calls of a unit whose continuations are of kinds return values of
   these kinds. *)
Lemma calls_kinded_of_results :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (ks0 : finite_map (bool * list flambda_kind))
     (ks : finite_map (list flambda_kind)),
     results_kinded kinds ks0 e = true ->
     (∀ (k : continuation) (l : list flambda_kind),
        lookup k ks = Some l -> ∃ b : bool, lookup k ks0 = Some (b, l)) ->
     calls_kinded ks e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (ks0 : finite_map (bool * list flambda_kind))
       (ks : finite_map (list flambda_kind)),
       handler_results_kinded kinds ks0 h = true ->
       (∀ (k : continuation) (l : list flambda_kind),
          lookup k ks = Some l -> ∃ b : bool, lookup k ks0 = Some (b, l)) ->
       handler_calls_kinded ks h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH kinds ks0 ks H Hks. exact (IH _ _ _ H Hks).
  - intros bindings body Hcodes IH kinds ks0 ks H Hks. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb].
    rewrite (IH _ _ _ Hb Hks), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *; try reflexivity.
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. simpl in Hbs |- *.
    apply (IHc _ _ [] Hbs). intros k l Hl. discriminate Hl.
  - intros k h body IHh IH kinds ks0 ks H Hks. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb].
    rewrite (IHh _ _ _ Hh Hks). simpl.
    apply (IH _ _ _ Hb). intros j l Hl. simpl in Hl |- *.
    destruct (Pos.eqb j k);
      [ injection Hl as <-; exists false; reflexivity | exact (Hks j l Hl) ].
  - intros handlers body IHh IH kinds ks0 ks H Hks. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite forallb_forall in Hhs.
    assert (Hks' : ∀ (j : continuation) (l : list flambda_kind),
              lookup j (map (fun '(k, h) => (k, handler_kinds h)) handlers
                        ++ ks) = Some l ->
              ∃ b : bool,
                lookup j (map (fun '(k, h) => (k, (false, handler_kinds h)))
                            handlers ++ ks0) = Some (b, l)).
    { intros j l Hl. rewrite (lookup_map_pairs_app handler_kinds) in Hl.
      assert (E : lookup j (map (fun '(k, h) => (k, (false, handler_kinds h)))
                              handlers ++ ks0)
                  = match lookup j handlers with
                    | Some h => Some (false, handler_kinds h)
                    | None => lookup j ks0
                    end)
        by exact (lookup_map_pairs_app (fun h => (false, handler_kinds h)) j
                    handlers ks0).
      rewrite E.
      destruct (lookup j handlers) as [h | ].
      - injection Hl as <-. exists false. reflexivity.
      - exact (Hks j l Hl). }
    apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hin.
      exact (IHh k h Hin _ _ _ (Hhs _ Hin) Hks').
    + exact (IH _ _ _ Hb Hks').
  - intros ap kinds ks0 ks H Hks. simpl in H |- *.
    destruct (lookup (ap_continuation ap) ks) as [l | ] eqn:Hl;
      [ | reflexivity ].
    destruct (Hks _ _ Hl) as (b & Hl0). rewrite Hl0 in H. exact H.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH kinds ks0 ks H Hks.
    exact (IH _ _ _ H Hks).
Qed.

(* A unit without calls meets [calls_kinded]. *)
Lemma calls_kinded_of_exact (codes : list (code_id * code)) :
  (∀ (e : expr) (ks : finite_map (list flambda_kind)),
     checked true codes e = true -> calls_kinded ks e = true)
  ∧ (∀ (h : continuation_handler) (ks : finite_map (list flambda_kind)),
       handler_checked true codes h = true ->
       handler_calls_kinded ks h = true).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH ks H. apply andb_true_iff in H as [_ H].
    exact (IH ks H).
  - intros bindings body _ IH ks H. apply andb_true_iff in H as [Hbs Hb].
    rewrite (IH ks Hb), andb_true_r. rewrite forallb_forall in Hbs |- *.
    intros b Hin. specialize (Hbs b Hin).
    destruct b; simpl in Hbs |- *; [ discriminate Hbs | reflexivity
                                   | reflexivity ].
  - intros k h body IHh IH ks H. apply andb_true_iff in H as [Hh Hb].
    rewrite (IHh ks Hh), (IH _ Hb). reflexivity.
  - intros handlers body IHh IH ks H. apply andb_true_iff in H as [Hhs Hb].
    rewrite (IH _ Hb), andb_true_r. rewrite forallb_forall in Hhs |- *.
    intros [k h] Hin. exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap ks H. discriminate H.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH ks H. exact (IH ks H).
Qed.

(* The code of a unit of the check is of the check, and its [Switch]es
   are on what the code id uses. *)
Lemma unit_codes_checked (G : graph) (ex : bool) (codes : list (code_id * code))
    (ff : positive) :
  (∀ e : expr,
     ∀ (ctx : option code_id) (xs : finite_map bool)
       (ks : finite_map (list flambda_kind)),
       scrutinees_used G ctx e -> checked ex codes e = true ->
       raises_kept xs e = true -> code_closed e = true ->
       calls_kinded ks e = true -> among ff (continuations_of e) = false ->
       ∀ (id : code_id) (c : code),
         (id, c) ∈ unit_codes e ->
         scrutinees_used G (Some id) (code_body c)
         ∧ checked ex codes (code_body c) = true
         ∧ raises_kept [] (code_body c) = true
         ∧ code_is_closed c = true
         ∧ calls_kinded [] (code_body c) = true
         ∧ among ff (continuations_of (code_body c)) = false)
  ∧ (∀ h : continuation_handler,
       ∀ (ctx : option code_id) (xs : finite_map bool)
         (ks : finite_map (list flambda_kind)),
         handler_scrutinees_used G ctx h -> handler_checked ex codes h = true ->
         handler_raises_kept xs h = true ->
         handler_code_closed h = true ->
         handler_calls_kinded ks h = true ->
         among ff (continuations_in_handler h) = false ->
         ∀ (id : code_id) (c : code),
           (id, c) ∈ handler_unit_codes h ->
           scrutinees_used G (Some id) (code_body c)
           ∧ checked ex codes (code_body c) = true
           ∧ raises_kept [] (code_body c) = true
           ∧ code_is_closed c = true
           ∧ calls_kinded [] (code_body c) = true
           ∧ among ff (continuations_of (code_body c)) = false).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH ctx xs ks Hs Hc Hd Hk Hkd Hf. simpl in Hc.
    apply andb_true_iff in Hc as [_ Hc].
    exact (IH ctx xs ks Hs Hc Hd Hk Hkd Hf).
  - intros bindings body _ IH ctx xs ks [Hsb Hs] Hc Hd Hk Hkd Hf id c Hin.
    simpl in Hc, Hd, Hk, Hin. cbn [calls_kinded] in Hkd.
    cbn [continuations_of] in Hf.
    apply andb_true_iff in Hc as [Hcb Hc].
    apply andb_true_iff in Hd as [Hdb Hd].
    apply andb_true_iff in Hk as [Hkb Hk].
    apply andb_true_iff in Hkd as [Hkdb Hkd].
    rewrite among_app in Hf. apply orb_false_iff in Hf as [Hfb Hf].
    apply in_app_or in Hin as [Hin | Hin];
      [ | exact (IH ctx xs ks Hs Hc Hd Hk Hkd Hf id c Hin) ].
    apply in_flat_map in Hin as (b & Hb & Hin).
    destruct b as [id' c' | id' | s const]; simpl in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= -> ->] | []].
    rewrite forallb_forall in Hcb, Hdb, Hkb, Hkdb.
    pose proof (Hcb _ Hb) as Hcb'. pose proof (Hdb _ Hb) as Hdb'.
    pose proof (Hkb _ Hb) as Hkb'. pose proof (Hkdb _ Hb) as Hkdb'.
    pose proof (among_flat_map_false _ _ _ _ Hfb Hb) as Hfb'.
    pose proof (proj1 (every_in _ _ _) Hsb _ Hb) as Hsb'.
    destruct c as [params mc ret exn md cbody].
    simpl in Hcb', Hdb', Hsb', Hkb', Hkdb' |- *.
    cbn [continuations_in_binding continuations_in_code] in Hfb'.
    rewrite among_cons, among_cons in Hfb'.
    apply orb_false_iff in Hfb' as [_ Hfb'].
    apply orb_false_iff in Hfb' as [_ Hfb'].
    apply andb_true_iff in Hcb' as [_ Hcb'].
    exact (conj Hsb' (conj Hcb' (conj Hdb' (conj Hkb' (conj Hkdb' Hfb'))))).
  - intros k h body IHh IH ctx xs ks [Hsh Hs] Hc Hd Hk Hkd Hf id c Hin.
    simpl in Hc, Hd, Hk, Hin. cbn [calls_kinded] in Hkd.
    cbn [continuations_of] in Hf.
    apply andb_true_iff in Hc as [Hch Hc].
    apply andb_true_iff in Hd as [Hdh Hd].
    apply andb_true_iff in Hk as [Hkh Hk].
    apply andb_true_iff in Hkd as [Hkdh Hkd].
    rewrite among_cons, among_app in Hf.
    apply orb_false_iff in Hf as [_ Hf]. apply orb_false_iff in Hf as [Hfh Hf].
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh ctx xs ks Hsh Hch Hdh Hkh Hkdh Hfh id c Hin)
      | exact (IH ctx _ _ Hs Hc Hd Hk Hkd Hf id c Hin) ].
  - intros handlers body IHh IH ctx xs ks [Hsh Hs] Hc Hd Hk Hkd Hf id c
      Hin.
    simpl in Hc, Hd, Hk, Hin. cbn [calls_kinded] in Hkd.
    cbn [continuations_of] in Hf.
    apply andb_true_iff in Hc as [Hch Hc].
    apply andb_true_iff in Hd as [Hdh Hd].
    apply andb_true_iff in Hk as [Hkh Hk].
    apply andb_true_iff in Hkd as [Hkdh Hkd].
    rewrite among_app in Hf. apply orb_false_iff in Hf as [Hfh Hf].
    rewrite forallb_forall in Hch, Hdh, Hkh, Hkdh.
    apply in_app_or in Hin as [Hin | Hin];
      [ | exact (IH ctx _ _ Hs Hc Hd Hk Hkd Hf id c Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh' & Hin).
    pose proof (among_flat_map_false _ _ _ _ Hfh Hkh') as Hfh'.
    cbn [fst snd] in Hfh'. rewrite among_cons in Hfh'.
    apply orb_false_iff in Hfh' as [_ Hfh'].
    exact (IHh k h Hkh' ctx _ _ (proj1 (every_in _ _ _) Hsh _ Hkh')
             (Hch _ Hkh') (Hdh _ Hkh') (Hkh _ Hkh') (Hkdh _ Hkh') Hfh' id c
             Hin).
  - intros ap ctx xs ks _ _ _ _ _ _ id c [].
  - intros ac ctx xs ks _ _ _ _ _ _ id c [].
  - intros scrutinee arms ctx xs ks _ _ _ _ _ _ id c [].
  - intros ctx xs ks _ _ _ _ _ _ id c [].
  - intros params metadata handler IH ctx xs ks Hs Hc Hd Hk Hkd Hf.
    exact (IH ctx xs ks Hs Hc Hd Hk Hkd Hf).
Qed.

(* [rebuild_refines] and [rebuild_refines_observably], in the two modes
   of the check. *)
(* ======================================================================== *)
(* The tuple of a call of tupled code is kept alive                         *)
(* ======================================================================== *)

Lemma add_code_dep_le (id : code_id) (cd : code_dep) (a : acc) :
  acc_le a (add_code_dep id cd a).
Proof.
  split; [ intros f H; exact H | ].
  split; [ intros s H; exact H | ].
  split; [ intros d H; exact H | ].
  intros x H. simpl. right. exact H.
Qed.

(* [prepare_code] of tupled code adds the facts of [tupled_kept], those
   of its witness of unknown arity and the two that keep its tuple
   alive. *)
Lemma prepare_code_keeps (id : code_id) (c : code) (a a' : acc)
    (cd : code_dep) :
  code_is_tupled c = true ->
  hd_error (acc_code_deps (prepare_code id c a)) = Some (id, cd) ->
  acc_le (prepare_code id c a) a' -> tupled_kept (acc_graph a') id cd.
Proof.
  intros Ht. unfold prepare_code.
  destruct (fresh_list (length (code_result_kinds c)) a) as [returns a1].
  cbv beta iota.
  destruct (fresh a1) as [exn a2]. cbv beta iota.
  destruct (fresh a2) as [mc a3]. cbv beta iota.
  destruct (fresh_list (length (code_params_kinds c)) a3) as [params a4].
  cbv beta iota.
  unfold create_known_arity_call_witness.
  destruct (fresh a4) as [w a5]. cbv beta iota.
  unfold create_unknown_arity_call_witnesses. rewrite Ht.
  unfold create_unknown_arity_tupled_call_witnesses.
  match goal with
  | |- context [fresh (add_facts ?l a5)] =>
    destruct (fresh (add_facts l a5)) as [W b1]
  end.
  cbv beta iota.
  destruct (fresh b1) as [U b2]. cbv beta iota.
  destruct (fresh b2) as [K b3]. cbv beta iota.
  intros Hhd. cbn [acc_code_deps add_code_dep hd_error] in Hhd.
  injection Hhd as <-.
  intros Hle.
  match type of Hle with
  | acc_le (add_code_dep _ _ (add_fact _ ?x)) _ =>
    assert (Hle7 : acc_le x a')
  end.
  { refine (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) _).
    exact (acc_le_trans _ _ _ (add_code_dep_le _ _ _) Hle). }
  exists W, U, K. cbn [cd_unknown_arity_call_witnesses cd_params].
  split; [ reflexivity | ].
  split; [ | split; [ | split ] ].
  - apply facts_parameter. apply (add_facts_later _ _ _ _ Hle7).
    apply in_or_app. right. right. right. left. reflexivity.
  - intros i p Hp. apply facts_accessor.
    apply (add_facts_later _ _ _ _ Hle7). apply in_or_app. right.
    right. right. right. apply in_or_app. right. apply in_or_app. left.
    apply in_map_iff.
    exists (i, p). split; [ reflexivity | exact (indexed_nth _ _ _ Hp) ].
  - apply facts_accessor. apply (add_facts_later _ _ _ _ Hle7).
    apply in_or_app. right. right. right. right. apply in_or_app. right.
    apply in_or_app. right. left. reflexivity.
  - apply facts_use. apply (add_facts_later _ _ _ _ Hle7).
    apply in_or_app. right. right. right. right. apply in_or_app. right.
    apply in_or_app. right. right. left. reflexivity.
Qed.

Lemma prepare_bindings_keeps (bindings : list static_binding) :
  ∀ (a : acc) (id : code_id) (c : code),
    Bind_code id c ∈ bindings -> code_is_tupled c = true ->
    ∃ cd : code_dep,
      (id, cd)
      ∈ acc_code_deps
          (List.fold_left (fun a b => prepare_binding b a) bindings a)
      ∧ ∀ a' : acc,
          acc_le (List.fold_left (fun a b => prepare_binding b a) bindings a)
            a' ->
          tupled_kept (acc_graph a') id cd.
Proof.
  induction bindings as [ | b bindings IH]; intros a id c Hin Ht;
    [ destruct Hin | ].
  destruct Hin as [-> | Hin];
    [ | exact (IH (prepare_binding b a) id c Hin Ht) ].
  cbn [List.fold_left prepare_binding].
  destruct (prepare_code_spec id c a) as (cd & Hcds & _).
  destruct (prepare_bindings_spec bindings (prepare_code id c a)) as [Hle _].
  exists cd. split.
  - apply (proj2 (proj2 (proj2 Hle))). rewrite Hcds. left. reflexivity.
  - intros a' Hle'. apply (prepare_code_keeps id c a a' cd Ht).
    + rewrite Hcds. reflexivity.
    + exact (acc_le_trans _ _ _ Hle Hle').
Qed.

(* The code deps of the tupled code of [l], with the facts of
   [tupled_kept]. *)
Definition kept_alive (l : list (code_id * code)) (a' : acc) : Prop :=
  ∀ (id : code_id) (c : code),
    (id, c) ∈ l -> code_is_tupled c = true ->
    ∃ cd : code_dep,
      (id, cd) ∈ acc_code_deps a' ∧ tupled_kept (acc_graph a') id cd.

Lemma kept_alive_app (l l' : list (code_id * code)) (a' : acc) :
  kept_alive l a' -> kept_alive l' a' -> kept_alive (l ++ l') a'.
Proof.
  intros H H' id c Hin. apply in_app_or in Hin as [Hin | Hin];
    [ exact (H id c Hin) | exact (H' id c Hin) ].
Qed.

Lemma fold_left_le {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (x : A) (a : acc), x ∈ l -> acc_le a (f a x)) ->
  ∀ a : acc, acc_le a (List.fold_left f l a).
Proof.
  induction l as [ | x l IH]; intros H a; simpl; [ apply acc_le_refl | ].
  exact (acc_le_trans _ _ _ (H x a (or_introl eq_refl))
           (IH (fun y b Hy => H y b (or_intror Hy)) (f a x))).
Qed.

Lemma fold_left_kept {A : Type} (f : acc -> A -> acc) (l : list A)
    (Q : A -> acc -> Prop) :
  (∀ (x : A) (a : acc), x ∈ l -> acc_le a (f a x)) ->
  (∀ (x : A) (a a' : acc), x ∈ l -> acc_le (f a x) a' -> Q x a') ->
  ∀ a a' : acc, acc_le (List.fold_left f l a) a' ->
    ∀ x : A, x ∈ l -> Q x a'.
Proof.
  induction l as [ | x l IH]; intros Hle HQ a a' Ha y Hy; [ destruct Hy | ].
  simpl in Ha. destruct Hy as [<- | Hy].
  - apply (HQ x a a' (or_introl eq_refl)).
    refine (acc_le_trans _ _ _ _ Ha).
    exact (fold_left_le f l (fun z b Hz => Hle z b (or_intror Hz)) (f a x)).
  - exact (IH (fun z b Hz => Hle z b (or_intror Hz))
             (fun z b b' Hz => HQ z b b' (or_intror Hz)) (f a x) a' Ha y Hy).
Qed.

(* Every later accumulator has the facts of [tupled_kept] of the tupled
   code of a term of the fragment. *)
Lemma traverse_keeps (codes : list (code_id * code)) (symbols : list symbol)
    (K : list value_slot) (slots_read : Prop) :
  (∀ e : expr,
     flow_fragment false codes symbols e = true ->
     distinct (static_names code_ids_of_binding e) = true ->
     (∀ w : value_slot, w ∈ projected_value_slots e -> among w K = true) ->
     slots_read ∨ projection_free e = true ->
     (∀ x : code_id * code, x ∈ unit_codes e -> x ∈ codes) ->
     ∀ env : traverse_env,
       env_value_slots_to_keep env = K ->
       ∀ a a' : acc,
         acc_le (traverse env e a) a' -> kept_alive (unit_codes e) a')
  ∧ (∀ h : continuation_handler,
       handler_flow_fragment false codes symbols h = true ->
       distinct (handler_static_names code_ids_of_binding h) = true ->
       (∀ w : value_slot,
          w ∈ handler_projected_value_slots h -> among w K = true) ->
       slots_read ∨ handler_projection_free h = true ->
       (∀ x : code_id * code, x ∈ handler_unit_codes h -> x ∈ codes) ->
       ∀ env : traverse_env,
         env_value_slots_to_keep env = K ->
         ∀ a a' : acc,
           acc_le (traverse_handler env h a) a' ->
           kept_alive (handler_unit_codes h) a').
Proof.
  destruct (traverse_makes codes symbols K slots_read false) as [Hmk Hmkh].
  apply expr_handler_ind.
  - (* [Let] *)
    intros x kind d body IH Hfrag Hdist Hproj Hslots Hcodes_in env HK a a'
      Hle.
    simpl in Hfrag, Hdist, Hproj, Hslots, Hcodes_in.
    apply andb_true_iff in Hfrag as [_ Hbody].
    exact (IH Hbody Hdist
             (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
             (slots_and_r _ _ _ Hslots) Hcodes_in env HK _ a' Hle).
  - (* [Let_static] *)
    intros bindings body _ IH Hfrag Hdist Hproj Hslots Hcodes_in env HK a a'
      Hle.
    simpl in Hfrag, Hdist, Hproj, Hslots, Hcodes_in.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    rewrite forallb_forall in Hbindings.
    pose proof (distinct_app_r _ _ Hdist) as Hdist_body.
    change (acc_le
              (traverse env body
                 (List.fold_left (fun a b => traverse_binding env b a)
                    bindings
                    (List.fold_left
                       (fun a b => traverse_static_binding env b a) bindings
                       (List.fold_left (fun a b => prepare_binding b a)
                          bindings a)))) a') in Hle.
    set (a_p := List.fold_left (fun a b => prepare_binding b a) bindings a)
      in *.
    set (a_s := List.fold_left (fun a b => traverse_static_binding env b a)
                  bindings a_p) in *.
    set (a_c := List.fold_left (fun a b => traverse_binding env b a)
                  bindings a_s) in *.
    assert (Hle_s : acc_le a_p a_s).
    { apply fold_left_le. intros b a0 Hin.
      specialize (Hbindings b Hin).
      destruct b as [id c | id | s [ | slot syms decls vs | t fields
                                   | contents | bytes | elements ]];
        simpl in Hbindings; try discriminate Hbindings;
        try apply acc_le_refl.
      - exact (proj1 (proj1 (static_set_makes env symbols s slot syms decls
                               vs Hbindings a0))).
      - exact (proj1 (proj1 (block_deps_makes (Node_symbol s) fields a0))).
      - exact (proj1 (proj1 (boxed_number_makes s contents a0))).
      - exact (proj1 (proj1 (alias_constants_makes (Node_symbol s) a0))).
      - destruct elements as [ | el els ];
          [ exact (proj1 (proj1 (alias_constants_makes (Node_symbol s) a0)))
          | ].
        exact (proj1 (proj1 (block_deps_makes (Node_symbol s) (el :: els)
                               a0))). }
    assert (Hle_c : acc_le a_s a_c).
    { apply fold_left_le. intros b a0 Hin.
      specialize (Hbindings b Hin).
      destruct b as [id c | id | s const]; try apply acc_le_refl.
      destruct c as [params mc ret exn metadata cbody].
      simpl in Hbindings.
      apply andb_true_iff in Hbindings as [Hb Hcb].
      apply andb_true_iff in Hb as [Hb Hcounted].
      apply andb_true_iff in Hb as [Hne Hbinds].
      cbn [traverse_binding traverse_code].
      destruct (lookup id (acc_code_deps a0)) as [cd | ];
        [ | apply acc_le_refl ].
      assert (Hnames : static_names code_ids_of_binding cbody = []).
      { apply (proj1 binds_statics_no_code_ids).
        apply negb_true_iff in Hbinds. exact Hbinds. }
      refine (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) _).
      exact (proj1 (proj1 (Hmk cbody Hcb
               ltac:(rewrite Hnames; reflexivity)
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_introl
                  (code_body_projected bindings id
                     (Code params mc ret exn metadata cbody) w Hin Hw))))
               (slots_forallb _ _ _ _ (slots_and_l _ _ _ Hslots) Hin)
               ltac:(intros x Hx;
                     rewrite (proj1 unit_codes_of_no_statics _
                                (proj1 (negb_true_iff _) Hbinds)) in Hx;
                     destruct Hx)
               (code_env env id ret exn cd) HK _))). }
    assert (Hle_b : acc_le a_c a').
    { refine (acc_le_trans _ _ _ _ Hle).
      exact (proj1 (proj1 (Hmk body Hbody Hdist_body
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
               (slots_and_r _ _ _ Hslots)
               (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
               env HK a_c))). }
    pose proof (acc_le_trans _ _ _ Hle_s (acc_le_trans _ _ _ Hle_c Hle_b))
      as Hle_pa.
    change (kept_alive
              (flat_map
                 (fun binding =>
                    match binding with
                    | Bind_code id c => [(id, c)]
                    | Bind_deleted_code _ | Bind_symbol _ _ => []
                    end)
                 bindings
               ++ unit_codes body) a').
    apply kept_alive_app.
    + intros id c Hin Ht.
      apply in_flat_map in Hin as (b & Hb & Hin).
      destruct b as [id' c' | id' | s const]; simpl in Hin;
        [ | destruct Hin | destruct Hin ].
      destruct Hin as [Heq | []]. injection Heq as -> ->.
      destruct (prepare_bindings_keeps bindings a id c Hb Ht)
        as (cd & Hcd & Hk).
      exists cd. split.
      * apply (proj2 (proj2 (proj2 Hle_pa))). exact Hcd.
      * exact (Hk a' Hle_pa).
    + exact (IH Hbody Hdist_body
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
               (slots_and_r _ _ _ Hslots)
               (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
               env HK a_c a' Hle).
  - (* [Let_cont_non_recursive] *)
    intros k h body IHh IH Hfrag Hdist Hproj Hslots Hcodes_in env HK a a'
      Hle.
    simpl in Hfrag, Hdist, Hproj, Hslots, Hcodes_in.
    apply andb_true_iff in Hfrag as [Hh Hbody].
    change (acc_le
              (traverse (add_cont env k (handler_params h)) body
                 (exn_handler_deps env h (traverse_handler env h a))) a')
      in Hle.
    change (kept_alive (handler_unit_codes h ++ unit_codes body) a').
    pose proof (proj1 (exn_handler_deps_steady env h
                         (traverse_handler env h a)))
      as Hle_x.
    pose proof (proj1 (proj1 (Hmk body Hbody (distinct_app_r _ _ Hdist)
                  (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
                  (slots_and_r _ _ _ Hslots)
                  (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
                  (add_cont env k (handler_params h)) HK
                  (exn_handler_deps env h (traverse_handler env h a)))))
      as Hle_b.
    apply kept_alive_app.
    + exact (IHh Hh (distinct_app_l _ _ Hdist)
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_introl Hw)))
               (slots_and_l _ _ _ Hslots)
               (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_introl Hx)))
               env HK a a'
               (acc_le_trans _ _ _ Hle_x (acc_le_trans _ _ _ Hle_b Hle))).
    + exact (IH Hbody (distinct_app_r _ _ Hdist)
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
               (slots_and_r _ _ _ Hslots)
               (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
               (add_cont env k (handler_params h)) HK _ a' Hle).
  - (* [Let_cont_recursive] *)
    intros handlers body IHh IH Hfrag Hdist Hproj Hslots Hcodes_in env HK a
      a' Hle.
    simpl in Hfrag, Hdist, Hproj, Hslots, Hcodes_in.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    set (env' := add_rec_conts env handlers).
    destruct (add_rec_conts_fields env handlers) as [_ HK'].
    assert (HK'' : env_value_slots_to_keep env' = K)
      by (unfold env'; rewrite HK'; exact HK).
    change (acc_le
              (traverse env' body
                 (List.fold_left
                    (fun a '(_, h) => traverse_handler env' h a) handlers a))
              a') in Hle.
    change (kept_alive
              (flat_map (fun '(_, h) => handler_unit_codes h) handlers
               ++ unit_codes body) a').
    assert (Hprem : ∀ (k : continuation) (h : continuation_handler),
              (k, h) ∈ handlers ->
              handler_flow_fragment false codes symbols h = true
              ∧ distinct (handler_static_names code_ids_of_binding h) = true
              ∧ (∀ w : value_slot,
                   w ∈ handler_projected_value_slots h -> among w K = true)
              ∧ (slots_read ∨ handler_projection_free h = true)
              ∧ (∀ x : code_id * code,
                   x ∈ handler_unit_codes h -> x ∈ codes)).
    { intros k h Hin. split; [ exact (Hhandlers _ Hin) | ].
      split;
        [ exact (distinct_flat_map _ _ _ (distinct_app_l _ _ Hdist) Hin)
        | ].
      split.
      { intros w Hw. apply Hproj. apply in_or_app. left.
        apply in_flat_map. exists (k, h). auto. }
      split; [ exact (slots_forallb _ _ _ _ (slots_and_l _ _ _ Hslots) Hin)
             | ].
      intros x Hx. apply Hcodes_in. apply in_or_app. left.
      apply in_flat_map. exists (k, h). auto. }
    assert (Hle_b :
      acc_le
        (List.fold_left (fun a '(_, h) => traverse_handler env' h a)
           handlers a)
        a').
    { refine (acc_le_trans _ _ _ _ Hle).
      exact (proj1 (proj1 (Hmk body Hbody (distinct_app_r _ _ Hdist)
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
               (slots_and_r _ _ _ Hslots)
               (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
               env' HK'' _))). }
    apply kept_alive_app.
    + intros id c Hin Ht.
      apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      refine (fold_left_kept (fun a '(_, h) => traverse_handler env' h a)
                handlers
                (fun '(_, h) a' => kept_alive (handler_unit_codes h) a')
                _ _ a a' Hle_b (k, h) Hkh id c Hin Ht).
      * intros [k0 h0] a0 Hin0.
        destruct (Hprem k0 h0 Hin0) as (H1 & H2 & H3 & H4 & H5).
        exact (proj1 (proj1 (Hmkh h0 H1 H2 H3 H4 H5 env' HK'' a0))).
      * intros [k0 h0] a0 a0' Hin0 Hle0.
        destruct (Hprem k0 h0 Hin0) as (H1 & H2 & H3 & H4 & H5).
        exact (IHh k0 h0 Hin0 H1 H2 H3 H4 H5 env' HK'' a0 a0' Hle0).
    + exact (IH Hbody (distinct_app_r _ _ Hdist)
               (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
               (slots_and_r _ _ _ Hslots)
               (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
               env' HK'' _ a' Hle).
  - (* [Apply] *)
    intros ap _ _ _ _ _ env _ a a' _ id c [].
  - (* [Apply_cont] *)
    intros ac _ _ _ _ _ env _ a a' _ id c [].
  - (* [Switch] *)
    intros scrutinee arms _ _ _ _ _ env _ a a' _ id c [].
  - (* [Invalid] *)
    intros _ _ _ _ _ env _ a a' _ id c [].
  - (* A handler *)
    intros params metadata handler IH Hfrag Hdist Hproj Hslots Hcodes_in env
      HK a a' Hle.
    exact (IH Hfrag Hdist Hproj Hslots Hcodes_in env HK a a' Hle).
Qed.

(* What [traverse_unit_facts] says of the graph of a unit of the fragment,
   and, of the code deps, that the tuple of each tupled code is kept
   alive. *)
Lemma traverse_unit_keeps (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (slots_read : Prop) :
  in_flow_fragment e = true ->
  slots_read ∨ projection_free e = true ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep),
    expr_edges (graph_of_facts (traverse_unit flag u e)) noted deps cds
      (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
      slots_read False None (unit_conts u) e
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e)) Node_all_constants
    ∧ (∀ s : symbol,
         noted s ->
         among s (static_names symbols_of_binding e) = false ->
         g_any_source (graph_of_facts (traverse_unit flag u e))
           (Node_symbol s))
    ∧ (∀ (k : continuation) (p : node),
         p ∈ unit_cont_nodes u k ->
         g_any_usage (graph_of_facts (traverse_unit flag u e)) p)
    ∧ (∀ d : closure_dep,
         deps d ->
         closure_dep_facts (graph_of_facts (traverse_unit flag u e)) cds d)
    ∧ NoDup (map fst cds)
    ∧ (∀ id : code_id,
         id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e)
    ∧ (∀ (id : code_id) (c : code) (cd : code_dep),
         (id, c) ∈ unit_codes e -> code_is_tupled c = true ->
         lookup id cds = Some cd ->
         tupled_kept (graph_of_facts (traverse_unit flag u e)) id cd)
    ∧ cds = unit_code_deps flag u e.
Proof.
  intros Hfrag Hslots. apply andb_true_iff in Hfrag as [Hfrag Hdist].
  rewrite traverse_unit_eq.
  destruct (proj1 (traverse_makes (unit_codes e)
                     (static_names symbols_of_binding e)
                     (projected_value_slots e) slots_read false) e Hfrag Hdist
              (fun w Hw => proj2 (among_In w _) Hw) Hslots (fun x Hx => Hx)
              (initial_env flag u e) eq_refl initial_acc)
    as [[Hle4 (l & Hl & Hperm)] Hedges].
  set (a4 := traverse (initial_env flag u e) e initial_acc) in *.
  set (a5 := add_fact (Fact_any_usage (Node_made 1)) a4).
  destruct (resolve_delayed_deps_spec (static_names symbols_of_binding e) a5)
    as (S6 & Hdeps6 & Hsyms6 & Hclosures & Himported).
  set (a6 := resolve_delayed_deps _ a5) in *.
  pose proof (steady_trans _ _ _ (add_facts_steady _ a4) S6) as S46.
  assert (Hcds : acc_code_deps a6 = l).
  { rewrite (proj2 S46), Hl, app_nil_r. reflexivity. }
  assert (Hnd : NoDup (map fst (acc_code_deps a6))).
  { rewrite Hcds.
    apply (Permutation.Permutation_NoDup (Permutation.Permutation_sym Hperm)).
    apply distinct_NoDup. exact Hdist. }
  exists (acc_noted a6), (acc_deps a6), (acc_code_deps a6).
  split; [ | split; [ | split; [ | split; [ | split; [ | split;
    [ | split; [ | split ] ] ] ] ] ] ].
  9: reflexivity.
  - exact (Hedges a6 (proj1 S46)).
  - apply facts_any_source. apply (proj1 S46), Hle4. right. left.
    reflexivity.
  - intros s Hs Hunit. apply facts_any_source.
    apply Himported; [ | exact Hunit ].
    unfold acc_noted in Hs. rewrite Hsyms6 in Hs. exact Hs.
  - intros k p Hp. unfold unit_cont_nodes, unit_conts in Hp.
    simpl in Hp. apply facts_any_usage.
    destruct (Pos.eqb k (unit_exn u));
      [ | destruct (Pos.eqb k (unit_return u)) ];
      simpl in Hp; [ | | contradiction ]; destruct Hp as [<- | []].
    + apply (proj1 S46), Hle4. left. reflexivity.
    + apply (proj1 S6). left. reflexivity.
  - intros d Hd. unfold acc_deps in Hd. rewrite Hdeps6 in Hd.
    pose proof (Hclosures d Hd) as H. rewrite <- (proj2 S6) in H.
    exact H.
  - exact Hnd.
  - intros id Hid. rewrite Hcds in Hid.
    exact (Permutation.Permutation_in _ Hperm Hid).
  - intros id c cd Hin Ht Hlookup.
    destruct (proj1 (traverse_keeps (unit_codes e)
                       (static_names symbols_of_binding e)
                       (projected_value_slots e) slots_read) e Hfrag Hdist
                (fun w Hw => proj2 (among_In w _) Hw) Hslots (fun x Hx => Hx)
                (initial_env flag u e) eq_refl initial_acc a6 (proj1 S46)
                id c Hin Ht)
      as (cd0 & Hin0 & Hk0).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin0) in Hlookup.
    injection Hlookup as <-. exact Hk0.
Qed.

(* The metadata of the code of a unit, where [static_names] looks. *)
Lemma bound_code_metadata_eq :
  (∀ e : expr,
     bound_code_metadata e
     = map (fun '(id, c) => (id, metadata_of_code c)) (unit_codes e))
  ∧ (∀ h : continuation_handler,
       handler_bound_code_metadata h
       = map (fun '(id, c) => (id, metadata_of_code c))
           (handler_unit_codes h)).
Proof.
  apply expr_handler_ind; try reflexivity.
  - intros x kind d body IH. exact IH.
  - intros bindings body _ IH. cbn [bound_code_metadata unit_codes].
    rewrite IH, map_app. f_equal.
    induction bindings as [ | b bs IHb]; [ reflexivity | ].
    cbn [flat_map]. rewrite map_app, IHb.
    destruct b; reflexivity.
  - intros k h body IHh IH. cbn [bound_code_metadata unit_codes].
    rewrite IHh, IH, map_app. reflexivity.
  - intros handlers body IHh IH. cbn [bound_code_metadata unit_codes].
    rewrite IH, map_app. f_equal.
    induction handlers as [ | [k h] hs IHhs]; [ reflexivity | ].
    cbn [flat_map]. rewrite map_app, (IHh k h (or_introl eq_refl)).
    f_equal. apply IHhs. intros k' h' Hin. apply (IHh k' h'). right.
    exact Hin.
  - intros params metadata handler IH. exact IH.
Qed.

(* The metadata of code of the unit, found by its code id, if the code
   ids are distinct. *)
Lemma lookup_bound_code_metadata (e : expr) (id : code_id) (c : code) :
  NoDup (map fst (unit_codes e)) -> (id, c) ∈ unit_codes e ->
  lookup id (bound_code_metadata e) = Some (metadata_of_code c).
Proof.
  intros Hnd Hin. rewrite (proj1 bound_code_metadata_eq e).
  revert Hnd Hin. generalize (unit_codes e). intros l.
  induction l as [ | [k c'] l IH]; intros Hnd Hin; [ destruct Hin | ].
  cbn [map lookup]. destruct (Pos.eqb_spec id k) as [-> | Hne].
  - destruct Hin as [[= <-] | Hin]; [ reflexivity | ].
    exfalso. inversion Hnd as [ | ? ? Hnotin _ ]. apply Hnotin.
    exact (in_map fst _ _ Hin).
  - destruct Hin as [[= -> _] | Hin]; [ contradiction | ].
    apply IH; [ inversion Hnd; assumption | exact Hin ].
Qed.

(* The code ids of the code of a unit, where [static_names] looks. *)
Lemma unit_codes_ids :
  (∀ e : expr, map fst (unit_codes e) = static_names code_ids_of_binding e)
  ∧ (∀ h : continuation_handler,
       map fst (handler_unit_codes h)
       = handler_static_names code_ids_of_binding h).
Proof.
  apply expr_handler_ind; try reflexivity.
  - intros x kind d body IH. exact IH.
  - intros bindings body _ IH. cbn [unit_codes static_names].
    rewrite map_app, IH. f_equal.
    induction bindings as [ | b bs IHb]; [ reflexivity | ].
    cbn [flat_map]. rewrite map_app, IHb. destruct b; reflexivity.
  - intros k h body IHh IH. cbn [unit_codes static_names].
    rewrite map_app, IHh, IH. reflexivity.
  - intros handlers body IHh IH. cbn [unit_codes static_names].
    rewrite map_app, IH. f_equal.
    induction handlers as [ | [k h] hs IHhs]; [ reflexivity | ].
    cbn [flat_map]. rewrite map_app, (IHh k h (or_introl eq_refl)).
    f_equal. apply IHhs. intros k' h' Hin. apply (IHh k' h'). right.
    exact Hin.
  - intros params metadata handler IH. exact IH.
Qed.

Lemma unit_codes_nodup (e : expr) :
  in_flow_fragment e = true -> NoDup (map fst (unit_codes e)).
Proof.
  intros Hfrag. apply andb_true_iff in Hfrag as [_ Hdist].
  rewrite (proj1 unit_codes_ids e). apply distinct_NoDup. exact Hdist.
Qed.

Lemma simples_of_kinds_sub (Kmap E : finite_map flambda_kind) :
  (∀ (x : variable) (k : flambda_kind),
     lookup x Kmap = Some k -> lookup x E = Some k) ->
  ∀ (args : list simple) (K : list flambda_kind),
    simples_of_kinds Kmap args K = true -> simples_of_kinds E args K = true.
Proof.
  intros Hsub. induction args as [ | a args IH]; intros [ | k K] H;
    cbn [simples_of_kinds] in H |- *; try exact H.
  destruct (simple_kind Kmap a) as [k' | ] eqn:Hk; [ | discriminate H ].
  assert (Hk' : simple_kind E a = Some k').
  { destruct a; cbn [simple_kind] in Hk |- *;
      [ exact (Hsub _ _ Hk) | exact Hk | exact Hk ]. }
  rewrite Hk'. apply andb_true_iff in H as [H1 H2]. rewrite H1.
  exact (IH K H2).
Qed.

Lemma kinds_sub_app (L Kmap E : finite_map flambda_kind) :
  (∀ (x : variable) (k : flambda_kind),
     lookup x Kmap = Some k -> lookup x E = Some k) ->
  ∀ (x : variable) (k : flambda_kind),
    lookup x (L ++ Kmap) = Some k -> lookup x (L ++ E) = Some k.
Proof.
  intros Hsub x k. rewrite !lookup_app.
  destruct (lookup x L); [ exact (fun H => H) | apply Hsub ].
Qed.

(* The kinds that [results_kinded] checks are those that
   FlambdaReaperCallingConventionsSpec.ret_calls_ok asks of what returns
   to the return continuation of code, at the variables of the rebuild,
   given the calls in tail position known. *)
Lemma ret_calls_ok_of_results (r : continuation) (K : list flambda_kind) :
  (∀ (e : expr) (Kmap E : finite_map flambda_kind)
     (ks : finite_map (bool * list flambda_kind)),
     results_kinded Kmap ks e = true ->
     (∀ (x : variable) (k : flambda_kind),
        lookup x Kmap = Some k -> lookup x E = Some k) ->
     lookup r ks = Some (true, K) ->
     ret_calls_ok E r K e = true)
  ∧ (∀ (h : continuation_handler) (Kmap E : finite_map flambda_kind)
       (ks : finite_map (bool * list flambda_kind)),
       handler_results_kinded Kmap ks h = true ->
       (∀ (x : variable) (k : flambda_kind),
          lookup x Kmap = Some k -> lookup x E = Some k) ->
       lookup r ks = Some (true, K) ->
       handler_ret_calls_ok E r K h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *.
    apply (IH _ _ ks Hr); [ | exact Hks ].
    intros y k. cbn [lookup].
    destruct (Pos.eqb y x); [ exact (fun H => H) | apply Hsub ].
  - intros bindings body _ IH Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *.
    apply andb_true_iff in Hr as [_ Hr]. exact (IH _ _ ks Hr Hsub Hks).
  - intros k h body IHh IH Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *.
    apply andb_true_iff in Hr as [Hrh Hrb].
    apply andb_true_intro. split; [ exact (IHh _ _ ks Hrh Hsub Hks) | ].
    destruct (Pos.eqb k r) eqn:Ekr; [ reflexivity | ].
    cbn [orb].
    apply (IH _ _ _ Hrb Hsub).
    cbn [lookup]. rewrite Pos.eqb_sym, Ekr. exact Hks.
  - intros handlers body IHh IH Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *.
    destruct (among r (map fst handlers)) eqn:Hin; [ reflexivity | ].
    cbn [orb].
    apply andb_true_iff in Hr as [Hrh Hrb].
    rewrite forallb_forall in Hrh.
    assert (Hks' : lookup r
                     (map (fun '(k, h) => (k, (false, handler_kinds h)))
                        handlers ++ ks)
                   = Some (true, K)).
    { rewrite lookup_app.
      destruct (lookup r (map (fun '(k, h) => (k, (false, handler_kinds h)))
                            handlers))
        as [x | ] eqn:Hl; [ | exact Hks ].
      exfalso. apply lookup_in in Hl.
      apply in_map_iff in Hl as ([k' h'] & [= <- _] & Hin').
      apply not_true_iff_false in Hin. apply Hin. apply among_In.
      exact (in_map fst _ (k', h') Hin'). }
    apply andb_true_intro. split.
    + apply forallb_forall. intros [k h] Hkh.
      exact (IHh k h Hkh _ _ _ (Hrh (k, h) Hkh) Hsub Hks').
    + exact (IH _ _ _ Hrb Hsub Hks').
  - intros ap Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *.
    destruct (Pos.eqb (ap_continuation ap) r) eqn:Er; [ | reflexivity ].
    apply Pos.eqb_eq in Er. cbn [negb orb].
    rewrite Er, Hks in Hr. cbn iota in Hr. exact Hr.
  - intros ac Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *. unfold returns_kinded in Hr.
    destruct (Pos.eqb (ac_continuation ac) r) eqn:Er; [ | reflexivity ].
    apply Pos.eqb_eq in Er. cbn [negb orb].
    rewrite Er, Hks in Hr. exact (simples_of_kinds_sub _ _ Hsub _ _ Hr).
  - intros scrutinee arms Kmap E ks Hr Hsub Hks.
    cbn [results_kinded ret_calls_ok] in Hr |- *.
    rewrite forallb_forall in Hr |- *.
    intros [i ac] Hin. specialize (Hr _ Hin). cbv beta iota in Hr |- *.
    unfold returns_kinded in Hr.
    destruct (Pos.eqb (ac_continuation ac) r) eqn:Er; [ | reflexivity ].
    apply Pos.eqb_eq in Er. cbn [negb orb].
    rewrite Er, Hks in Hr. exact (simples_of_kinds_sub _ _ Hsub _ _ Hr).
  - intros. reflexivity.
  - intros params metadata handler IH Kmap E ks Hr Hsub Hks.
    cbn [handler_results_kinded handler_ret_calls_ok] in Hr |- *.
    destruct (Nat.eqb (length params) (length (hm_params_kinds metadata)))
      eqn:Hl; [ | reflexivity ].
    apply Nat.eqb_eq in Hl. cbn [negb orb].
    rewrite (with_kinds_combine params _ (eq_sym Hl)).
    exact (IH _ _ ks Hr (kinds_sub_app _ _ _ Hsub) Hks).
Qed.

(* The code of a unit whose exception handlers are well formed has them
   well formed. *)
Lemma unit_codes_exn_wf :
  (∀ e : expr,
     exn_handlers_well_formed e = true ->
     ∀ (id : code_id) (c : code),
       (id, c) ∈ unit_codes e -> code_exn_handlers_well_formed c = true)
  ∧ (∀ h : continuation_handler,
       handler_exn_handlers_well_formed h = true ->
       ∀ (id : code_id) (c : code),
         (id, c) ∈ handler_unit_codes h ->
         code_exn_handlers_well_formed c = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Hw. exact (IH Hw).
  - intros bindings body _ IH Hw id c Hin.
    cbn [exn_handlers_well_formed unit_codes] in Hw, Hin.
    apply andb_true_iff in Hw as [Hb Hw]. rewrite forallb_forall in Hb.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH Hw id c Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id' c' | id' | s const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= <- <-] | []]. exact (Hb _ Hbin).
  - intros k h body IHh IH Hw id c Hin.
    cbn [exn_handlers_well_formed unit_codes] in Hw, Hin.
    apply andb_true_iff in Hw as [Hw Hwb].
    apply andb_true_iff in Hw as [_ Hwh].
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh Hwh id c Hin) | exact (IH Hwb id c Hin) ].
  - intros handlers body IHh IH Hw id c Hin.
    cbn [exn_handlers_well_formed unit_codes] in Hw, Hin.
    apply andb_true_iff in Hw as [Hwh Hwb]. rewrite forallb_forall in Hwh.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH Hwb id c Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    pose proof (Hwh (k, h) Hkh) as H. cbv beta iota in H.
    apply andb_true_iff in H as [_ H].
    exact (IHh k h Hkh H id c Hin).
  - intros ap Hw id c [].
  - intros ac Hw id c [].
  - intros scrutinee arms Hw id c [].
  - intros Hw id c [].
  - intros params metadata handler IH Hw id c Hin. exact (IH Hw id c Hin).
Qed.

(* The code of a unit whose results are kinded has its results kinded. *)
Lemma unit_codes_results_kinded :
  (∀ (e : expr) (Kmap : finite_map flambda_kind)
     (ks : finite_map (bool * list flambda_kind)),
     results_kinded Kmap ks e = true ->
     ∀ (id : code_id) (c : code),
       (id, c) ∈ unit_codes e -> code_results_kinded c = true)
  ∧ (∀ (h : continuation_handler) (Kmap : finite_map flambda_kind)
       (ks : finite_map (bool * list flambda_kind)),
       handler_results_kinded Kmap ks h = true ->
       ∀ (id : code_id) (c : code),
         (id, c) ∈ handler_unit_codes h -> code_results_kinded c = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Kmap ks Hr. exact (IH _ _ Hr).
  - intros bindings body _ IH Kmap ks Hr id c Hin.
    cbn [results_kinded unit_codes] in Hr, Hin.
    apply andb_true_iff in Hr as [Hb Hr]. rewrite forallb_forall in Hb.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH _ _ Hr id c Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id' c' | id' | s const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= <- <-] | []]. exact (Hb _ Hbin).
  - intros k h body IHh IH Kmap ks Hr id c Hin.
    cbn [results_kinded unit_codes] in Hr, Hin.
    apply andb_true_iff in Hr as [Hrh Hrb].
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh _ _ Hrh id c Hin) | exact (IH _ _ Hrb id c Hin) ].
  - intros handlers body IHh IH Kmap ks Hr id c Hin.
    cbn [results_kinded unit_codes] in Hr, Hin.
    apply andb_true_iff in Hr as [Hrh Hrb]. rewrite forallb_forall in Hrh.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH _ _ Hrb id c Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    exact (IHh k h Hkh _ _ (Hrh (k, h) Hkh) id c Hin).
  - intros ap Kmap ks _ id c [].
  - intros ac Kmap ks _ id c [].
  - intros scrutinee arms Kmap ks _ id c [].
  - intros Kmap ks _ id c [].
  - intros params metadata handler IH Kmap ks Hr id c Hin.
    exact (IH _ _ Hr id c Hin).
Qed.

Lemma code_change_all_unchanged (sol : solution)
    (code_deps : finite_map code_dep) (ids : list code_id) (id : code_id)
    (md : code_metadata) (pd rd : list param_decision) :
  code_calling_convention_change sol (fun _ => true) code_deps ids id md
  <> Changing_calling_convention pd rd.
Proof.
  unfold code_calling_convention_change, cannot_change_calling_convention.
  rewrite orb_true_r. discriminate.
Qed.

Lemma rets_nil_bound (r : continuation) (K : list flambda_kind) :
  lookup r ([] : finite_map (list flambda_kind)) = Some K ->
  among r [] = false.
Proof. discriminate. Qed.

Lemma rets_nil_any (P : Prop) (r : continuation) (K : list flambda_kind) :
  lookup r ([] : finite_map (list flambda_kind)) = Some K -> P.
Proof. discriminate. Qed.

(* The code of a unit that is well formed and whose results are kinded:
   no call raises to its return continuation, and what returns there is
   of the kinds of its result arity. *)
Lemma code_rets_of_unit (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> results_kinded [] [] e = true ->
  ∀ (id : code_id) (params : list variable) (mc : variable)
    (ret exn : continuation) (md : code_metadata) (body : expr),
    (id, Code params mc ret exn md body) ∈ unit_codes e ->
    (ret <> exn -> no_raise_to ret body = true)
    ∧ (length params = length (unarize (cm_params_arity md)) ->
       ret_calls_ok
         (rev (with_kinds params (unarize (cm_params_arity md)))
          ++ [(mc, Value)])
         ret (unarize (cm_result_arity md)) body
       = true).
Proof.
  intros Hwf Hrk id params mc ret exn md body Hin.
  split.
    { (* From the ways out of the code, which are well formed. *)
      intros Hne.
      assert (Hue : unit_exn_handlers_well_formed (unit_exn u) e = true).
      { unfold well_formed_unit in Hwf.
        destruct (unit_exn_handlers_well_formed (unit_exn u) e);
          [ reflexivity | ].
        rewrite andb_false_r in Hwf. discriminate Hwf. }
      unfold unit_exn_handlers_well_formed in Hue.
      apply andb_true_iff in Hue as [_ Hewf].
      pose proof (proj1 unit_codes_exn_wf e Hewf id _ Hin) as Hcw.
      cbn [code_exn_handlers_well_formed] in Hcw.
      apply andb_true_iff in Hcw as [Hcw _].
      apply andb_true_iff in Hcw as [Hways _].
      destruct (no_raise_to ret body) eqn:E; [ reflexivity | ].
      pose proof E as E0. apply (proj1 (free_of_raise ret)) in E.
      unfold ways_out_well_formed in Hways.
      rewrite forallb_forall in Hways. specialize (Hways ret E).
      rewrite (proj2 (Pos.eqb_neq ret exn) Hne) in Hways.
      apply andb_true_iff in Hways as [Hways _].
      apply negb_true_iff in Hways.
      rewrite (proj1 (no_raise_of_raises_to ret) body Hways) in E0.
      discriminate E0. }
  intros Lpar.
  pose proof (proj1 unit_codes_results_kinded e [] [] Hrk id _ Hin)
    as Hcrk.
  cbn [code_results_kinded] in Hcrk.
  apply (proj1 (ret_calls_ok_of_results ret (unarize (cm_result_arity md)))
           body _ _ _ Hcrk).
  - rewrite (with_kinds_combine params _ (eq_sym Lpar)).
    intros x k H. exact H.
  - cbn [lookup]. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma rebuild_refines_checked (ex : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  checked ex (unit_codes e) e = true ->
  calls_kinded [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  (ex = false -> results_kinded [] [] e = true) ->
  program_refines_observably e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e)
  ∧ (ex = true ->
     program_refines e
       (rebuild_unit_from_tables (traverse_unit flag u e) T
          preserve_direct_calls e)).
Proof.
  intros Hlocal Hslots Hfrag Hcf Hkinded Hwf Hexits Hcheck Hcheck_exact
    Hrk_ex.
  rename preserve_direct_calls into b.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  pose proof (check_field_used_exact_sound is_local _ _ Hcheck_exact)
    as Hexact.
  destruct (traverse_unit_keeps flag u e (∀ f : field, is_local f = false)
              Hfrag Hslots)
    as (noted & deps & cds & He & Hconsts & Himported & Hunit & Hdeps & Hnd
        & Hkeys & Hkeep_alive & _).
  pose proof (traverse_unit_scrutinees ex (unit_codes e) flag u e Hcf)
    as Hscr.
  set (F := traverse_unit flag u e) in *.
  set (G := graph_of_facts F) in *.
  set (sol := solution_of_tables T) in *.
  assert (Hclosed : code_closed e = true).
  { unfold well_formed_unit in Hwf.
    destruct (code_closed e); [ reflexivity | ].
    rewrite andb_false_r in Hwf. discriminate Hwf. }
  assert (Hfresh : among (fresh_name (names_of e)) (continuations_of e)
                   = false).
  { apply not_true_iff_false. intros E. apply among_In in E.
    pose proof (fresh_name_fresh (names_of e)) as Hff.
    apply not_true_iff_false in Hff. apply Hff. apply among_In.
    unfold names_of. apply in_or_app. right. exact E. }
  assert (Hagree0 : kinds_agree [] (env_conts FlambdaReaperRebuild.empty_env)
                      [] []).
  { intros k m Hl. discriminate Hl. }
  (* The calls raise to exception handlers, with extra arguments only to
     handlers in scope. *)
  assert (Hraises : raises_kept [] e = true).
  { assert (Hue : unit_exn_handlers_well_formed (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (unit_exn_handlers_well_formed (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold unit_exn_handlers_well_formed in Hue.
    apply andb_true_iff in Hue as [Hue Hewf].
    apply andb_true_iff in Hue as [Hways _].
    apply (proj1 raises_kept_of_well_formed e [] Hewf);
      [ intros k Hl; discriminate Hl | ].
    intros k _ Hk. unfold ways_out_well_formed in Hways.
    rewrite forallb_forall in Hways. specialize (Hways k Hk).
    apply andb_true_iff in Hways as [_ Hg]. apply negb_true_iff in Hg.
    exact Hg. }
  assert (Hxagree0 : exn_agree (env_conts FlambdaReaperRebuild.empty_env)
                       []).
  { intros k. exact I. }
  pose proof (proj1 (unit_codes_checked G ex (unit_codes e)
                       (fresh_name (names_of e))) e None
                [] [] Hscr Hcf Hraises Hclosed Hkinded Hfresh)
    as Hcode_checks.
  (* What FlambdaReaperFlowProofs.flow_sound derives of the graph. *)
  pose proof (static_names_edges G noted deps cds (unit_kept e) (unit_codes e)
                (static_names symbols_of_binding e)
                (∀ f : field, is_local f = false) False) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst cds (id, cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact Hw. }
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs. apply among_In in Hs.
    destruct (Hsym_names s Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id | ] eqn:Hl; [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - exact (boxed_self is_local G sol Hc Hconsts _ _ Hb).
    - exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
    - destruct elements as [ | el els ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))). }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit_s;
      [ exact (Hunit_syms s Hunit_s) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit_s)). }
  (* The continuations that the unit names are its own. *)
  assert (Hfree : ∀ k : continuation,
            k ∈ free_continuations_of_expr e -> unit_cont u k).
  { assert (Hnames_in : names_in_scope (unit_return u) (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (names_in_scope (unit_return u) (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold names_in_scope, names_among in Hnames_in.
    apply andb_true_iff in Hnames_in as [_ Hks].
    intros k Hk.
    pose proof (proj1 (all_among _ _) Hks k (proj2 (among_In _ _) Hk)) as Hin.
    apply among_In in Hin. destruct Hin as [<- | [<- | []]];
      [ left | right ]; reflexivity. }
  assert (Hcc0 : cc_closed G sol (fun _ => true) (fun _ => true))
    by (constructor; intros; reflexivity).
  assert (Hcode_rets :
    ∀ (id : code_id) (params : list variable) (mc : variable)
      (ret exn : continuation) (md : code_metadata) (body : expr),
      ex = false ->
      (id, Code params mc ret exn md body) ∈ unit_codes e ->
      (ret <> exn -> no_raise_to ret body = true)
      ∧ (length params = length (unarize (cm_params_arity md)) ->
         ret_calls_ok
           (rev (with_kinds params (unarize (cm_params_arity md)))
            ++ [(mc, Value)])
           ret (unarize (cm_result_arity md)) body
         = true))
    by (intros id params mc ret exn md body Hex;
        exact (code_rets_of_unit u e Hwf (Hrk_ex Hex) id params mc ret exn md
                 body)).
  unfold program_refines, rebuild_unit_from_tables, rebuild_unit.
  (* The rebuild where no code changes its calling convention. *)
  rewrite <- (rebuild_expr_cc_unchanged F sol (fun _ => true) [] b
                (static_names code_ids_bound e) (bound_code_metadata e)
                (fresh_name (names_of e)) (fun _ _ => eq_refl)).
  eapply (unit_refines is_local G sol noted deps cds (unit_kept e)
            (unit_codes e) (static_names symbols_of_binding e) u F b);
    eassumption || reflexivity || (intros; reflexivity)
    || (intros; discriminate)
    || (intros ? ? ? ? ? ? ? ? ? _ Hch;
        destruct (code_change_all_unchanged _ _ _ _ _ _ _ Hch))
    || exact (conj I (conj (ex_intro _ []
                              (conj (conj (fun k H => H) rets_nil_bound)
                                 (exits_bound_unit u e Hexits Hfree)))
                            (conj Hscr (conj Hcf (conj (conj Hfresh
                               (conj (ex_intro _ [] (conj Hkinded Hagree0))
                                  (ex_intro _ [] (conj Hraises Hxagree0))))
                               (fun r K H => rets_nil_any _ r K H)))))).
Qed.

(* The same for the rebuild with changes of calling convention, given
   what the checks of the table say of it. *)
Lemma rebuild_cc_refines_checked (ex : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change cannot_change_witness : node -> bool)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  checked ex (unit_codes e) e = true ->
  calls_kinded [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  results_kinded [] [] e = true ->
  cc_closed (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    cannot_change_witness cannot_change ->
  (∀ a a' n y : node,
     g_accessor (graph_of_facts (traverse_unit flag u e)) a
       (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a'
       (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source (solution_of_tables T) a' = true ->
      s_any_source (solution_of_tables T) a = true)
     ∧ (s_sources (solution_of_tables T) a' y = true ->
        s_sources (solution_of_tables T) a y = true)) ->
  (∀ (w : node) (id id' : code_id),
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id) ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id') ->
     id' = id) ->
  program_refines_observably e
    (rebuild_unit_cc (traverse_unit flag u e) (solution_of_tables T)
       cannot_change (unit_code_deps flag u e) preserve_direct_calls e)
  ∧ (ex = true ->
     program_refines e
       (rebuild_unit_cc (traverse_unit flag u e) (solution_of_tables T)
          cannot_change (unit_code_deps flag u e) preserve_direct_calls e)).
Proof.
  intros Hlocal Hslots Hfrag Hcf Hkinded Hwf Hexits Hcheck Hcheck_exact
    Hrk Hccc Hwid Hcid.
  rename preserve_direct_calls into b.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  pose proof (check_field_used_exact_sound is_local _ _ Hcheck_exact)
    as Hexact.
  destruct (traverse_unit_keeps flag u e (∀ f : field, is_local f = false)
              Hfrag Hslots)
    as (noted & deps & cds & He & Hconsts & Himported & Hunit & Hdeps & Hnd
        & Hkeys & Hkeep_alive & Hcds_eq).
  pose proof (traverse_unit_scrutinees ex (unit_codes e) flag u e Hcf)
    as Hscr.
  set (F := traverse_unit flag u e) in *.
  set (G := graph_of_facts F) in *.
  set (sol := solution_of_tables T) in *.
  assert (Hclosed : code_closed e = true).
  { unfold well_formed_unit in Hwf.
    destruct (code_closed e); [ reflexivity | ].
    rewrite andb_false_r in Hwf. discriminate Hwf. }
  assert (Hfresh : among (fresh_name (names_of e)) (continuations_of e)
                   = false).
  { apply not_true_iff_false. intros E. apply among_In in E.
    pose proof (fresh_name_fresh (names_of e)) as Hff.
    apply not_true_iff_false in Hff. apply Hff. apply among_In.
    unfold names_of. apply in_or_app. right. exact E. }
  assert (Hagree0 : kinds_agree [] (env_conts FlambdaReaperRebuild.empty_env)
                      [] []).
  { intros k m Hl. discriminate Hl. }
  (* The calls raise to exception handlers, with extra arguments only to
     handlers in scope. *)
  assert (Hraises : raises_kept [] e = true).
  { assert (Hue : unit_exn_handlers_well_formed (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (unit_exn_handlers_well_formed (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold unit_exn_handlers_well_formed in Hue.
    apply andb_true_iff in Hue as [Hue Hewf].
    apply andb_true_iff in Hue as [Hways _].
    apply (proj1 raises_kept_of_well_formed e [] Hewf);
      [ intros k Hl; discriminate Hl | ].
    intros k _ Hk. unfold ways_out_well_formed in Hways.
    rewrite forallb_forall in Hways. specialize (Hways k Hk).
    apply andb_true_iff in Hways as [_ Hg]. apply negb_true_iff in Hg.
    exact Hg. }
  assert (Hxagree0 : exn_agree (env_conts FlambdaReaperRebuild.empty_env)
                       []).
  { intros k. exact I. }
  pose proof (proj1 (unit_codes_checked G ex (unit_codes e)
                       (fresh_name (names_of e))) e None
                [] [] Hscr Hcf Hraises Hclosed Hkinded Hfresh)
    as Hcode_checks.
  (* What FlambdaReaperFlowProofs.flow_sound derives of the graph. *)
  pose proof (static_names_edges G noted deps cds (unit_kept e) (unit_codes e)
                (static_names symbols_of_binding e)
                (∀ f : field, is_local f = false) False) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst cds (id, cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact Hw. }
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs. apply among_In in Hs.
    destruct (Hsym_names s Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id | ] eqn:Hl; [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - exact (boxed_self is_local G sol Hc Hconsts _ _ Hb).
    - exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
    - destruct elements as [ | el els ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))). }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit_s;
      [ exact (Hunit_syms s Hunit_s) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit_s)). }
  (* The continuations that the unit names are its own. *)
  assert (Hfree : ∀ k : continuation,
            k ∈ free_continuations_of_expr e -> unit_cont u k).
  { assert (Hnames_in : names_in_scope (unit_return u) (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (names_in_scope (unit_return u) (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold names_in_scope, names_among in Hnames_in.
    apply andb_true_iff in Hnames_in as [_ Hks].
    intros k Hk.
    pose proof (proj1 (all_among _ _) Hks k (proj2 (among_In _ _) Hk)) as Hin.
    apply among_In in Hin. destruct Hin as [<- | [<- | []]];
      [ left | right ]; reflexivity. }
  (* What the rebuild with changes of calling convention reads. *)
  pose proof (code_rets_of_unit u e Hwf Hrk) as Hcode_rets0.
  assert (Hcode_rets :
    ∀ (id : code_id) (params : list variable) (mc : variable)
      (ret exn : continuation) (md : code_metadata) (body : expr),
      ex = false ->
      (id, Code params mc ret exn md body) ∈ unit_codes e ->
      (ret <> exn -> no_raise_to ret body = true)
      ∧ (length params = length (unarize (cm_params_arity md)) ->
         ret_calls_ok
           (rev (with_kinds params (unarize (cm_params_arity md)))
            ++ [(mc, Value)])
           ret (unarize (cm_result_arity md)) body
         = true))
    by (intros id params mc ret exn md body _; apply Hcode_rets0).
  assert (Hcode_deps :
    ∀ id : code_id,
      cannot_change (Node_code_id id) = false ->
      lookup id (unit_code_deps flag u e) = lookup id cds)
    by (intros id _; rewrite Hcds_eq; reflexivity).
  assert (Hmetadatas :
    ∀ (id : code_id) (c : code),
      (id, c) ∈ unit_codes e -> cannot_change (Node_code_id id) = false ->
      lookup id (bound_code_metadata e) = Some (metadata_of_code c))
    by (intros id c Hin _;
        exact (lookup_bound_code_metadata e id c (unit_codes_nodup e Hfrag)
                 Hin)).
  assert (Hwid' :
    ∀ (id : code_id) (a a' n y : node),
      cannot_change (Node_code_id id) = false ->
      g_accessor G a (Field_call_witness Known_arity_code_pointer) n ->
      g_accessor G a' (Field_call_witness Known_arity_code_pointer) n ->
      (s_any_source sol a' = true -> s_any_source sol a = true)
      ∧ (s_sources sol a' y = true -> s_sources sol a y = true))
    by (intros id a a' n y _; apply Hwid).
  assert (Hcid' :
    ∀ (id id' : code_id) (w : node),
      cannot_change (Node_code_id id) = false ->
      g_constructor G w Field_code_id_of_call_witness (Node_code_id id) ->
      g_constructor G w Field_code_id_of_call_witness (Node_code_id id') ->
      id' = id)
    by (intros id id' w _; apply Hcid).
  unfold program_refines, program_refines_observably, rebuild_unit_cc.
  eapply (unit_refines is_local G sol noted deps cds (unit_kept e)
            (unit_codes e) (static_names symbols_of_binding e) u F b);
    eassumption || reflexivity || (intros; reflexivity)
    || exact (conj I (conj (ex_intro _ []
                              (conj (conj (fun k H => H) rets_nil_bound)
                                 (exits_bound_unit u e Hexits Hfree)))
                            (conj Hscr (conj Hcf (conj (conj Hfresh
                               (conj (ex_intro _ [] (conj Hkinded Hagree0))
                                  (ex_intro _ [] (conj Hraises Hxagree0))))
                               (fun r K H => rets_nil_any _ r K H)))))).
Qed.

(* The check of the proof, if exact, is that of the statement. *)
Lemma checked_modes (codes : list (code_id * code)) :
  (∀ e : expr, checked true codes e = no_code_no_slots e)
  ∧ (∀ h : continuation_handler,
       handler_checked true codes h = handler_no_code_no_slots h).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH. rewrite IH. reflexivity.
  - intros bindings body _ IH. rewrite IH. f_equal.
    induction bindings as [ | b bindings IHb]; [ reflexivity | ].
    simpl. rewrite IHb. f_equal. destruct b; reflexivity.
  - intros k h body IHh IH. rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH. rewrite IH. f_equal.
    induction handlers as [ | [k h] handlers IHl]; [ reflexivity | ].
    simpl. rewrite (IHh k h (or_introl eq_refl)).
    rewrite IHl; [ reflexivity | ].
    intros k' h' Hin. exact (IHh k' h' (or_intror Hin)).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH. exact IH.
Qed.

(* The check of the statement up to observation is that of the proof. *)
Lemma static_constants_checked (codes : list (code_id * code)) :
  (∀ e : expr,
     static_constants codes e = true -> checked false codes e = true)
  ∧ (∀ h : continuation_handler,
       handler_static_constants codes h = true ->
       handler_checked false codes h = true).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | reflexivity | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap _. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* So is that of the statement with arguments of any kind. *)
Lemma any_kind_arguments_checked (codes : list (code_id * code)) :
  (∀ e : expr, any_kind_arguments e = true -> checked false codes e = true)
  ∧ (∀ h : continuation_handler,
       handler_any_kind_arguments h = true ->
       handler_checked false codes h = true).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | reflexivity | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap _. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem no_code_no_call : no_code_no_call_statement.
Proof. exact (proj1 no_code_no_call_all). Qed.

Theorem no_code_no_slots_no_code : no_code_no_slots_no_code_statement.
Proof.
  enough (H : (∀ e : expr, no_code_no_slots e = true -> no_code e = true)
              ∧ (∀ h : continuation_handler,
                   handler_no_code_no_slots h = true ->
                   handler_no_code h = true))
    by exact (proj1 H).
  apply expr_handler_ind.
  - intros x kind d body IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hd Hb]. rewrite (IH Hb), andb_true_r.
    destruct d; simpl in Hd |- *; try discriminate Hd; reflexivity.
  - intros bindings body _ IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [ | | s [ | slot syms decls [ | ] | | | | ]];
      simpl in Hbs |- *; try discriminate Hbs; try reflexivity; exact Hbs.
  - intros k h body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - intros ac H. exact H.
  - intros scrutinee arms H. exact H.
  - intros H. exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* A unit without calls meets [direct_tail_calls]. *)
Theorem no_call_direct_tail_calls : no_call_direct_tail_calls_statement.
Proof.
  enough (H : (∀ (e : expr) (bound : list continuation),
                 no_call e = true -> direct_tail_calls bound e = true)
              ∧ (∀ (h : continuation_handler) (bound : list continuation),
                   handler_no_call h = true ->
                   handler_direct_tail_calls bound h = true))
    by (intros e; exact (proj1 H e [])).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH bound H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH _ Hb).
  - intros bindings body Hcodes IH bound H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | reflexivity | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. simpl in Hbs.
    exact (IHc [] (proj1 no_code_no_call_all _ Hbs)).
  - intros k h body IHh IH bound H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh _ Hh), (IH _ Hb).
    reflexivity.
  - intros handlers body IHh IH bound H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap bound H. discriminate H.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH bound H. exact (IH _ H).
Qed.

(* A unit of direct tail calls meets [direct_calls]. *)
Theorem direct_tail_calls_direct_calls :
  direct_tail_calls_direct_calls_statement.
Proof.
  enough (H : (∀ (e : expr) (bound : list continuation),
                 direct_tail_calls bound e = true ->
                 direct_calls bound e = true)
              ∧ (∀ (h : continuation_handler) (bound : list continuation),
                   handler_direct_tail_calls bound h = true ->
                   handler_direct_calls bound h = true))
    by (intros bound e; exact (proj1 H e bound)).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH bound H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH _ Hb).
  - intros bindings body Hcodes IH bound H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc [] Hbs).
  - intros k h body IHh IH bound H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh _ Hh), (IH _ Hb).
    reflexivity.
  - intros handlers body IHh IH bound H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap bound H. unfold direct_tail_call in H. unfold direct_call.
    destruct (ap_call_kind ap); [ | discriminate H ].
    apply andb_true_iff in H as [H Hk]. apply andb_true_iff in H as [H _].
    apply andb_true_iff in H as [H Hx]. apply andb_true_iff in H as [_ He].
    rewrite He, Hx, Hk. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH bound H. exact (IH _ H).
Qed.

(* A unit of [direct_calls] meets [only_direct_calls]. *)
Theorem direct_calls_only_direct_calls :
  direct_calls_only_direct_calls_statement.
Proof.
  enough (H : (∀ (e : expr) (bound : list continuation),
                 direct_calls bound e = true -> only_direct_calls e = true)
              ∧ (∀ (h : continuation_handler) (bound : list continuation),
                   handler_direct_calls bound h = true ->
                   handler_only_direct_calls h = true))
    by (intros bound e; exact (proj1 H e bound)).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH bound H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH _ Hb).
  - intros bindings body Hcodes IH bound H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc [] Hbs).
  - intros k h body IHh IH bound H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh _ Hh), (IH _ Hb).
    reflexivity.
  - intros handlers body IHh IH bound H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap bound H. unfold direct_call in H. unfold only_direct_call.
    destruct (ap_call_kind ap); [ | discriminate H ].
    apply andb_true_iff in H as [_ Hk]. exact Hk.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH bound H. exact (IH _ H).
Qed.

Theorem only_direct_calls_full_calls :
  only_direct_calls_full_calls_statement.
Proof.
  intros codes.
  enough (H : (∀ e : expr,
                 only_direct_calls e = true -> full_calls codes e = true)
              ∧ (∀ h : continuation_handler,
                   handler_only_direct_calls h = true ->
                   handler_full_calls codes h = true))
    by exact (proj1 H).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. unfold only_direct_call in H. unfold full_call.
    destruct (ap_call_kind ap); [ | discriminate H ].
    rewrite H. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem full_calls_generic_calls : full_calls_generic_calls_statement.
Proof.
  intros codes.
  enough (H : (∀ e : expr,
                 full_calls codes e = true -> generic_calls codes e = true)
              ∧ (∀ h : continuation_handler,
                   handler_full_calls codes h = true ->
                   handler_generic_calls codes h = true))
    by exact (proj1 H).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. unfold full_call in H. unfold generic_call.
    apply andb_true_iff in H as [Hk Hcs]. rewrite Hk. cbn [andb].
    destruct (ap_call_kind ap); [ reflexivity | ].
    rewrite forallb_forall in Hcs |- *. intros [id c] Hin.
    specialize (Hcs (id, c) Hin). cbv beta iota in Hcs |- *.
    apply andb_true_iff in Hcs as [Hcs _]. exact Hcs.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem generic_calls_tupled_calls : generic_calls_tupled_calls_statement.
Proof.
  intros codes.
  enough (H : (∀ e : expr,
                 generic_calls codes e = true -> tupled_calls codes e = true)
              ∧ (∀ h : continuation_handler,
                   handler_generic_calls codes h = true ->
                   handler_tupled_calls codes h = true))
    by exact (proj1 H).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. unfold generic_call in H. unfold tupled_call.
    apply andb_true_iff in H as [Hk Hcs]. rewrite Hk. cbn [andb].
    destruct (ap_call_kind ap); [ reflexivity | ].
    rewrite forallb_forall in Hcs |- *. intros [id c] Hin.
    specialize (Hcs (id, c) Hin). cbv beta iota in Hcs |- *.
    rewrite Hcs. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem tupled_calls_static_constants :
  tupled_calls_static_constants_statement.
Proof.
  intros codes.
  enough (H : (∀ e : expr,
                 tupled_calls codes e = true ->
                 static_constants codes e = true)
              ∧ (∀ h : continuation_handler,
                   handler_tupled_calls codes h = true ->
                   handler_static_constants codes h = true))
    by exact (proj1 H).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | ].
    + pose proof (Hcodes id c Hin) as IHc.
      destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
    + destruct const; try exact Hbs; discriminate Hbs.
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem static_constants_any_kind_arguments :
  static_constants_any_kind_arguments_statement.
Proof.
  intros codes.
  enough (H : (∀ e : expr,
                 static_constants codes e = true ->
                 any_kind_arguments e = true)
              ∧ (∀ h : continuation_handler,
                   handler_static_constants codes h = true ->
                   handler_any_kind_arguments h = true))
    by exact (proj1 H).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH H.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd. exact (IH Hb).
  - intros bindings body Hcodes IH H.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ | exact Hbs | exact Hbs ].
    pose proof (Hcodes id c Hin) as IHc.
    destruct c as [params mc ret exn md cbody]. exact (IHc Hbs).
  - intros k h body IHh IH H.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap _. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem rebuild_refines : rebuild_refines_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hcf Hwf Hexits Hcheck
    Hcheck_exact.
  pose proof (closure_free_no_code_no_slots e Hcf) as Hns.
  refine (proj2 (rebuild_refines_checked true flag is_local u e T b Hlocal
                   Hslots Hfrag _ _ Hwf Hexits Hcheck Hcheck_exact
                   (fun E => match E with end)) eq_refl).
  - rewrite (proj1 (checked_modes (unit_codes e)) e). exact Hns.
  - apply (proj1 (calls_kinded_of_exact (unit_codes e))).
    rewrite (proj1 (checked_modes (unit_codes e)) e). exact Hns.
Qed.

Theorem example : example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ intros f; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { intros T Hin. vm_compute in Hin.
    repeat (destruct Hin as [<- | Hin]; [ vm_compute; reflexivity | ]).
    destruct Hin. }
  split; [ vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem exits_one_value_needed : exits_one_value_needed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State TwoValuesExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 5)]
         empty_heap)).
  { exists 2; try exists no_reads. vm_compute. reflexivity. }
  assert (Htarget :
    behaves (State TwoValuesExample.rebuilt [] [] empty_static_env
               empty_heap no_reads)
      (Exits 1%positive
         [Value_const (Tagged_immediate 1); Value_const (Poison Value)]
         empty_heap)).
  { exists 1; try exists no_reads. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  inversion Heq.
Qed.

Theorem field_used_exact_needed : field_used_exact_needed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State UnreadFieldExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 0)] empty_heap)).
  { exists 5; try exists no_reads. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State UnreadFieldExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      Goes_wrong).
  { exists 1; try exists no_reads. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  discriminate Heq.
Qed.

Theorem closure_exit_counterexample : closure_exit_counterexample_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State ClosureExitExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
            [(60%positive, Value_const (Tagged_immediate 5))]]
         empty_heap)).
  { exists 4; try exists no_reads. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State ClosureExitExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []]
         empty_heap)).
  { exists 3; try exists no_reads. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  inversion Heq.
Qed.

Theorem wrapper_kinds_counterexample :
  wrapper_kinds_counterexample_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State WrapperKindsExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 0)] empty_heap)).
  { exists 6; try exists no_reads. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State WrapperKindsExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      Goes_wrong).
  { exists 6; try exists no_reads. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  discriminate Heq.
Qed.

(* ======================================================================== *)
(* Refinement up to observation                                             *)
(* ======================================================================== *)

Theorem refines_observably_of_refines :
  refines_observably_of_refines_statement.
Proof.
  intros s t [Hexits Hends]. split; [ | split ].
  - intros k vs hp Hs. exists vs.
    split; [ exact (Hexits k vs hp Hs) | apply values_observed_refl ].
  - intros k vs' hp Ht.
    destruct (Hends (Exits k vs' hp) ltac:(discriminate) Ht) as [Hs | Hw].
    + left. exists vs'. split; [ exact Hs | apply values_observed_refl ].
    + right. exact Hw.
  - intros Ht.
    destruct (Hends Goes_wrong ltac:(discriminate) Ht) as [Hs | Hw];
      assumption.
Qed.

Theorem rebuild_refines_observably : rebuild_refines_observably_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact.
  assert (Hkinded : calls_kinded [] e = true).
  { apply (proj1 calls_kinded_of_results e [] [] [] Hrk).
    intros k l Hl. discriminate Hl. }
  exact (proj1 (rebuild_refines_checked false flag is_local u e T b Hlocal
                  Hslots Hfrag
                  (proj1 (any_kind_arguments_checked (unit_codes e)) e Hdc)
                  Hkinded Hwf Hexits Hcheck Hcheck_exact (fun _ => Hrk))).
Qed.

Theorem closure_exit_observed : closure_exit_observed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  assert (Hs :
    behaves
      (State ClosureExitExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
            [(60%positive, Value_const (Tagged_immediate 5))]]
         empty_heap))
    by (exists 4; try exists no_reads; vm_compute; reflexivity).
  assert (Ht :
    behaves
      (State ClosureExitExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []]
         empty_heap))
    by (exists 3; try exists no_reads; vm_compute; reflexivity).
  assert (Hobs :
    values_related observed_equal
      [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
         [(60%positive, Value_const (Tagged_immediate 5))]]
      [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []])
    by (cbn; repeat split).
  unfold program_refines_observably, refines_observably.
  split; [ | split ].
  - intros k vs hp Hb. pose proof (behaviour_unique _ _ _ Hs Hb) as Heq.
    injection Heq as <- <- <-. eexists. split; [ exact Ht | exact Hobs ].
  - intros k vs' hp Hb. pose proof (behaviour_unique _ _ _ Ht Hb) as Heq.
    injection Heq as <- <- <-. left. eexists. split; [ exact Hs | exact Hobs ].
  - intros Hw. pose proof (behaviour_unique _ _ _ Ht Hw) as Heq.
    discriminate Heq.
Qed.

Theorem results_kinded_needed : results_kinded_needed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  intros (_ & _ & Hwrong).
  assert (Hs :
    behaves
      (State WrapperKindsExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 0)] empty_heap))
    by (exists 6; try exists no_reads; vm_compute; reflexivity).
  assert (Ht :
    behaves
      (State WrapperKindsExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      Goes_wrong)
    by (exists 6; try exists no_reads; vm_compute; reflexivity).
  pose proof (behaviour_unique _ _ _ Hs (Hwrong Ht)) as Heq.
  discriminate Heq.
Qed.

Theorem closures_example : closures_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem value_slots_example : value_slots_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 12; try exists no_reads; vm_compute; reflexivity | ].
  split; [ exists 12; try exists no_reads; vm_compute; reflexivity | ].
  split.
  - unfold ValueSlotsExample.source_result, ValueSlotsExample.rebuilt_result,
      ValueSlotsExample.result.
    cbn. repeat split.
  - intros H. vm_compute in H. discriminate H.
Qed.

Theorem code_example : code_example_statement.
Proof.
  do 7 (split; [ vm_compute; reflexivity | ]).
  vm_compute. reflexivity.
Qed.

Theorem call_exits_one_value_needed : call_exits_one_value_needed_statement.
Proof.
  do 7 (split; [ vm_compute; reflexivity | ]).
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State CallExitsExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_const (Tagged_immediate 7); Value_const (Tagged_immediate 8)]
         empty_heap)).
  { exists 10; try exists no_reads. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State CallExitsExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive
         [Value_const (Tagged_immediate 7); Value_const (Poison Value)]
         empty_heap)).
  { exists 10; try exists no_reads. vm_compute. reflexivity. }
  destruct (Hexits _ _ _ Hsource) as (vs' & Ht & Hrel).
  pose proof (behaviour_unique _ _ _ Ht Htarget) as Heq.
  injection Heq as Hvs. subst vs'.
  simpl in Hrel. destruct Hrel as [_ [H _]]. discriminate H.
Qed.

Theorem tail_call_example : tail_call_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem wrapper_example : wrapper_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 12; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem raise_example : raise_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem indirect_example : indirect_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem partial_exit_counterexample :
  partial_exit_counterexample_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 10; try exists no_reads; vm_compute; reflexivity | ].
  split; [ exists 10; try exists no_reads; vm_compute; reflexivity | ].
  split.
  - cbn. intros [[_ [[H _] _]] _]. discriminate H.
  - cbn. repeat split.
Qed.

Theorem partial_call_example : partial_call_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem tupled_call_example : tupled_call_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem static_constants_example : static_constants_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem static_boxed_variable_example :
  static_boxed_variable_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem region_argument_example : region_argument_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem region_projection_needed : region_projection_needed_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  intros (_ & _ & Hwrong).
  assert (Hs :
    behaves
      (State RegionProjectionExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 7)] empty_heap))
    by (exists 5; try exists no_reads; vm_compute; reflexivity).
  assert (Ht :
    behaves
      (State RegionProjectionExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      Goes_wrong)
    by (exists 5; try exists no_reads; vm_compute; reflexivity).
  pose proof (behaviour_unique _ _ _ Hs (Hwrong Ht)) as Heq.
  discriminate Heq.
Qed.

(* ======================================================================== *)
(* Refinement up to observation, with the heap                              *)
(* ======================================================================== *)

Lemma values_related_impl (R R' : value -> value -> Prop) (l l' : list value) :
  (∀ v v' : value, v ∈ l -> R v v' -> R' v v') ->
  values_related R l l' -> values_related R' l l'.
Proof.
  revert l'. induction l as [ | v l IH]; intros [ | v' l'] H; simpl; auto.
  intros [Hv Hl]. split; [ exact (H v v' (or_introl eq_refl) Hv) | ].
  apply IH; [ | exact Hl ]. intros w w' Hw. exact (H w w' (or_intror Hw)).
Qed.

(* Observed equal values are observed equal with each location related to
   itself. *)
Lemma observed_equal_at_eq :
  ∀ v v' : value, observed_equal v v' -> observed_equal_at eq v v'.
Proof.
  intros v. pattern v. apply FlambdaHeapProofs.value_induction; clear v;
    intros; destruct v'; cbn [observed_equal observed_equal_at] in *;
    try contradiction; try assumption.
  - destruct H0 as [Htag Hfields]. split; [ exact Htag | ].
    revert Hfields. apply values_related_impl. intros w w' Hw.
    rewrite Forall_forall in H. exact (H w Hw w').
  - revert H0. apply values_related_impl. intros w w' Hw.
    rewrite Forall_forall in H. exact (H w Hw w').
  - destruct H1 as (Hc & Hk & Ha). split; [ exact (H _ Hc) | ].
    split; assumption.
Qed.

Lemma observed_equal_at_refl : ∀ v : value, observed_equal_at eq v v.
Proof. intros v. apply observed_equal_at_eq, observed_equal_refl. Qed.

Lemma values_observed_at_refl (l : list value) :
  values_related (observed_equal_at eq) l l.
Proof.
  induction l as [ | v l IH]; simpl; [ exact I | ].
  split; [ apply observed_equal_at_refl | exact IH ].
Qed.

(* Equal heaps are observed equal with each location related to
   itself. *)
Lemma heaps_observed_equal_refl (hp : heap) : heaps_observed_equal eq hp hp.
Proof.
  split; [ intros l l1 l2 <- <-; reflexivity | ].
  split; [ intros l1 l2 l' -> <-; reflexivity | ].
  intros l l' <-. destruct (lookup l hp) as [[mut tag fields | kind els] | ];
    simpl; [ | | exact I ].
  - split; [ reflexivity | split; [ reflexivity | ] ].
    apply values_observed_at_refl.
  - split; [ reflexivity | apply values_observed_at_refl ].
Qed.

Lemma exits_observed_equal_of (vs vs' : list value) (hp : heap) :
  values_related observed_equal vs vs' -> exits_observed_equal vs hp vs' hp.
Proof.
  intros H. exists eq. split; [ apply heaps_observed_equal_refl | ].
  revert H. apply values_related_impl. intros v v' _.
  apply observed_equal_at_eq.
Qed.

Theorem refines_up_to_heap_of_refines_observably :
  refines_up_to_heap_of_refines_observably_statement.
Proof.
  intros source target (Hexits & Hback & Hwrong).
  split; [ | split; [ | exact Hwrong ] ].
  - intros k vs hp Hb. destruct (Hexits k vs hp Hb) as (vs' & Ht & Hrel).
    exists vs', hp. split; [ exact Ht | exact (exits_observed_equal_of _ _ _
        Hrel) ].
  - intros k vs' hp Hb.
    destruct (Hback k vs' hp Hb) as [(vs & Hs & Hrel) | Hw]; [ left | right ].
    + exists vs, hp. split; [ exact Hs | exact (exits_observed_equal_of _ _ _
        Hrel) ].
    + exact Hw.
Qed.

Theorem heap_rebuild_example : heap_rebuild_example_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  assert (Hs :
    behaves
      (State FlowHeapExample.unit_expr [] [] empty_static_env empty_heap
         no_reads)
      (Exits 1%positive [FlowHeapExample.result]
         FlowHeapExample.heap_at_exit))
    by (exists 8; try exists no_reads; vm_compute; reflexivity).
  assert (Ht :
    behaves
      (State HeapRebuildExample.rebuilt [] [] empty_static_env empty_heap
         no_reads)
      (Exits 1%positive [HeapRebuildExample.result]
         HeapRebuildExample.heap_at_exit))
    by (exists 7; try exists no_reads; vm_compute; reflexivity).
  split.
  - (* The heaps differ: no exit of the rebuilt unit has the unit's. *)
    intros (Hexits & _ & _).
    destruct (Hexits _ _ _ Hs) as (vs' & Hb & _).
    pose proof (behaviour_unique _ _ _ Ht Hb) as Heq.
    injection Heq as _ Hheap. vm_compute in Hheap. discriminate Hheap.
  - (* The location 3 of the first run is the location 2 of the
       second. *)
    assert (Hobs :
      exits_observed_equal [FlowHeapExample.result]
        FlowHeapExample.heap_at_exit [HeapRebuildExample.result]
        HeapRebuildExample.heap_at_exit).
    { exists (fun l l' => l = 3%positive ∧ l' = 2%positive). split.
      - split; [ intros l l1 l2 [_ ->] [_ ->]; reflexivity | ].
        split; [ intros l1 l2 l' [-> _] [-> _]; reflexivity | ].
        intros l l' [-> ->]. vm_compute.
        split; [ reflexivity | split; [ reflexivity | split; [ reflexivity
                                                            | exact I ] ] ].
      - vm_compute. repeat split. }
    unfold program_refines_up_to_heap, refines_up_to_heap.
    split; [ | split ].
    + intros k vs hp Hb. pose proof (behaviour_unique _ _ _ Hs Hb) as Heq.
      injection Heq as <- <- <-. do 2 eexists. split; [ exact Ht | exact Hobs ].
    + intros k vs' hp' Hb. pose proof (behaviour_unique _ _ _ Ht Hb) as Heq.
      injection Heq as <- <- <-. left. do 2 eexists.
      split; [ exact Hs | exact Hobs ].
    + intros Hw. pose proof (behaviour_unique _ _ _ Ht Hw) as Heq.
      discriminate Heq.
Qed.
