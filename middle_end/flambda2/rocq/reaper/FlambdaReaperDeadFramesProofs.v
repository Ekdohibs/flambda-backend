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

(* The proofs of FlambdaReaperDeadFramesSpec.v. Nothing here needs review.

   The run of the unit keeps the invariant of FlambdaReaperFlowProofs.v
   ([flow_invariant], with no primitive of the heap and no location: [inv],
   [inv_run]), from which a variable with no source is never bound
   ([inv_unbound]). The call of the statement is entered as the flow proof
   enters a direct call ([direct_call_ok]), with its return continuation
   the renamed handler ([marked_call]). That handler is a continuation of
   the invariant at the results of the code: at a rank other than [i] what
   the code returns is read into the parameter of the handler that the
   call returns to, as in the flow proof; at the rank [i] nothing is, as
   the result has no source ([marker_ok]). The invariant is kept by the
   run from there, so [x] is never bound ([marked_run]).

   A call that [apply_never_returns] makes one that never returns reads,
   at the node [a], the call witness of known arity of its callee, which
   may come from the allocation [b] of the closure called; the witness [w]
   of the code dep is among the sources of [a] ([never_returns_result]).
   If the code is known, [w] gives [a] its code id, so the code is the one
   [code_id_actually_called] finds; otherwise [w] is among the witnesses
   that [never_returns_of_call] reads, and the result of [w] it finds with
   no source is one of the code dep. The arity of the call is that of the
   code, as the call steps ([entered_dead_result]).

   [dead_return_dropped]: the body of the code has the edges of the flow
   with its return continuation at the results of the code dep
   ([unit_codes_edges]); an [Apply_cont] to it, not under a binding of it,
   aliases each result to its argument ([applies_to_edges]), so that an
   argument with a source would give one to the result.

   Calls of unknown arity, the run only ([marked_indirect_run]): the call
   is entered as the flow proof enters it ([call_ok]), with its return
   continuation the renamed handler at the nodes of the results of the
   call, rank [i] replaced by [x]. That is a continuation of the
   invariant if nothing reaches the results of rank [i] of what the last
   witness of the call stands for ([dead_rank], [pending_marked]). Not
   shown here: that [never_returns_of_call] gives [call_dead_rank]. It
   reads the witnesses of the sources of the callee; [call_dead_rank]
   asks it of every source of the call's own witness, to which tables
   that are only closed may give more sources. With a least solution and
   a check of the facts, FlambdaReaperIndirectDeadFramesProofs.v shows
   it. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaReaperCallingConventions FlambdaReaperDeadFramesSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperLocalFieldsSpec FlambdaReaperLocalFieldsProofs
  FlambdaReaperWrapperProofs.
From OxCamlFlambda2 Require Import FlambdaReaperRebuild
  FlambdaReaperUnboxing.
From OxCamlFlambda2 Require FlambdaReaperLeast FlambdaReaperLeastSpec
  FlambdaReaperLeastProofs FlambdaReaperUnboxingRefinesCallsSpec.
Import ListNotations.
Import FlambdaReaperFlowProofs.WithLocal.

(* ======================================================================== *)
(* Renaming a parameter                                                     *)
(* ======================================================================== *)

Lemma rename_nth_other (i j : nat) (x : variable) (l : list variable) :
  j <> i -> nth_error (rename_nth i x l) j = nth_error l j.
Proof.
  revert i j. induction l as [ | y l IH]; intros i j Hji;
    [ destruct i, j; reflexivity | ].
  destruct i as [ | i], j as [ | j]; simpl;
    [ contradiction | reflexivity | reflexivity | ].
  apply IH. lia.
Qed.

Lemma rename_nth_at (i : nat) (x y : variable) (l : list variable) :
  nth_error (rename_nth i x l) i = Some y -> y = x.
Proof.
  revert i. induction l as [ | z l IH]; intros i H;
    [ destruct i; discriminate H | ].
  destruct i as [ | i]; simpl in H; [ congruence | exact (IH i H) ].
Qed.

Lemma rename_nth_length (i : nat) (x : variable) (l : list variable) :
  length (rename_nth i x l) = length l.
Proof.
  revert i. induction l as [ | z l IH]; intros i; [ destruct i; reflexivity | ].
  destruct i as [ | i]; simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma resolve_head (k : continuation) (c : cont_value) (conts : cont_env) :
  resolve k ((k, c) :: conts) = c.
Proof. unfold resolve. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma resolve_other (k k' : continuation) (c : cont_value)
    (conts : cont_env) :
  k <> k' -> resolve k ((k', c) :: conts) = resolve k conts.
Proof.
  intros Hne. unfold resolve. simpl.
  destruct (Pos.eqb_spec k k') as [-> | _]; [ contradiction | reflexivity ].
Qed.

(* ======================================================================== *)
(* The run                                                                  *)
(* ======================================================================== *)

Section Frames.

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
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
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
Hypothesis Hlrs : FlambdaReaperLocalFieldsSpec.local_reads_used is_local G sol.

(* The flow fragment has no primitive of the heap, and makes no location. *)
Definition no_location : location -> option node := fun _ => None.

Definition inv : state -> Prop :=
  flow_invariant G sol noted deps cds keep codes symbols u False no_location
    True.

Abbreviation flows_ := (flows G sol keep (called_by G cds) no_location).
Abbreviation cont_ok_ :=
  (cont_ok G sol noted deps cds keep codes symbols u False no_location True).

Lemma inv_step (s s' : state) : inv s -> step s = Step s' -> inv s'.
Proof.
  intros Hs Hstep.
  pose proof (step_flow_invariant is_local G sol noted deps cds keep codes
                symbols u False no_location True Hc Hlocal Hconsts Hsyms
                Hunit_syms Hdeps Hnd Hcds (fun _ => Hlrs) s (fun H => H)
                (fun _ => eq_refl) Hs) as H.
  rewrite Hstep in H. exact H.
Qed.

Lemma inv_run (m : nat) : ∀ s s' : state, inv s -> run m s = Step s' -> inv s'.
Proof.
  induction m as [ | m IH]; intros s s' Hs Hrun.
  - simpl in Hrun. injection Hrun as <-. exact Hs.
  - rewrite run_succ in Hrun.
    destruct (step s) as [s1 | | ] eqn:E; try discriminate Hrun.
    exact (IH s1 s' (inv_step s s1 Hs E) Hrun).
Qed.

(* A variable with no source is never bound. *)
Lemma inv_unbound (s : state) (x : variable) :
  inv s -> s_has_source sol (Node_variable x) = false ->
  lookup x (state_vars s) = None.
Proof.
  intros Hs Hx.
  destruct (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hs)
    as (Hvars & _ & _).
  destruct (lookup x (state_vars s)) as [v | ] eqn:E; [ | reflexivity ].
  pose proof (Hvars x v E) as Hf.
  rewrite (flows_hs is_local G sol keep (called_by G cds) no_location Hc v _
             Hf) in Hx.
  discriminate Hx.
Qed.

(* The handler that a call returns to, its parameter of rank [i] renamed,
   is a continuation at the nodes [ns'] of the results of the code: at
   every other rank what is there goes to the node of the call, and at
   [i] nothing is. *)
Lemma marker_ok (h : continuation_handler) (vars0 : env) (K0 : cont_env)
    (rec0 : list (continuation * continuation_handler)) (ns ns' : list node)
    (i : nat) (x : variable) :
  cont_ok_ (Closure h vars0 K0 rec0) ns ->
  (∀ (j : nat) (n' n : node) (v : value),
     j <> i -> nth_error ns' j = Some n' -> nth_error ns j = Some n ->
     flows_ v n' -> flows_ v n) ->
  length ns' = length ns ∨ length (handler_params h) <= length ns ->
  (∀ (n' : node) (v : value),
     nth_error ns' i = Some n' -> flows_ v n' -> False) ->
  cont_ok_ (Closure (rename_param i x h) vars0 K0 rec0) ns'.
Proof.
  intros Hok Htr Hlen Hdead.
  destruct (cont_ok_closure_inv _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hok)
    as (ctx & ce0 & Hvars & Hconts & Hlive & Hrec & Hh & Hcount & Ht).
  apply (Closure_ok _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ctx ce0);
    try assumption; [ destruct h; exact Hh | destruct h; exact Hcount | ].
  destruct h as [params md body].
  unfold handler_params in Ht, Hlen |- *. simpl in Ht, Hlen |- *.
  intros j v n' p Hn' Hp Hv.
  destruct (Nat.eq_dec j i) as [-> | Hji].
  - exfalso. exact (Hdead n' v Hn' Hv).
  - rewrite nth_error_map, rename_nth_other in Hp by exact Hji.
    assert (Hj : j < length ns).
    { assert (j < length (map Node_variable params))
        by (apply nth_error_Some; rewrite nth_error_map; congruence).
      assert (j < length ns') by (apply nth_error_Some; congruence).
      rewrite length_map in *. destruct Hlen; lia. }
    destruct (nth_error ns j) as [n | ] eqn:Hn;
      [ | apply nth_error_None in Hn; lia ].
    refine (Ht j v n p Hn _ (Htr j n' n v Hji Hn' Hn Hv)).
    rewrite nth_error_map. exact Hp.
Qed.

(* The call of known arity [ap] of [s], of a closure of the code [id] whose
   result of rank [i] has no source, made with its continuation the handler
   it returns to with its parameter of rank [i] renamed: the code is entered
   with the invariant. *)
Lemma marked_call (s : state) (ap : apply_expr) (callee : value)
    (id : code_id) (slots : list (value_slot * value)) (cd : code_dep)
    (i : nat) (r : node) (h : continuation_handler) (vars0 : env)
    (K0 : cont_env) (rec0 : list (continuation * continuation_handler))
    (x : variable) :
  inv s -> state_expr s = Apply ap ->
  ap_call_kind ap <> Indirect_unknown_arity ->
  ap_exn_continuation ap <> ap_continuation ap ->
  eval_simple (state_vars s) (ap_callee ap) = Some callee ->
  closure_of (state_statics s) callee = Some (id, slots) ->
  lookup id cds = Some cd -> nth_error (cd_return cd) i = Some r ->
  s_has_source sol r = false ->
  resolve (ap_continuation ap) (state_conts s) = Closure h vars0 K0 rec0 ->
  match
    step
      (State (Apply ap) (state_vars s)
         ((ap_continuation ap, Closure (rename_param i x h) vars0 K0 rec0)
          :: state_conts s)
         (state_statics s) (state_heap s) (state_reads s))
  with
  | Step s1 => inv s1
  | _ => True
  end.
Proof.
  intros Hs Hexpr Hnu Hexn_ne Hcallee Hclosure Hcd Hr Hrs Hres.
  destruct s as [e0 vars K statics hp rd]. simpl in *. subst e0.
  destruct Hs as (ctx & ce & He & Hlive & Hcount & Hvars & Hconts & Hst
                  & Hcodes & Hpa).
  destruct Hpa as (Hvars_pa & Hspa & Hconts_pa). simpl in *.
  destruct He as (Hfrag & Hnoted_callee & Hnoted_args & returns & exn & Hret
                  & Hexn & Hedges).
  set (marker := Closure (rename_param i x h) vars0 K0 rec0).
  pose proof (full_call_step G noted deps cds keep codes symbols False True ap
                vars ((ap_continuation ap, marker) :: K) statics hp rd Hnu
                Hfrag Hcodes) as Hfull.
  destruct (step _) as [s1 | | ]; [ | contradiction | exact I ].
  destruct Hfull as (id' & callee' & vs & extra & slots' & c & callee_vars
                     & _ & Hcallee' & Hvs & Hextra & Hclosure' & Hcode & _ & _
                     & _ & Hk1 & Hk2 & Hk3 & Hbind & ->).
  rewrite Hcallee in Hcallee'. injection Hcallee' as <-.
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  rewrite resolve_head.
  rewrite (resolve_other _ _ _ _ Hexn_ne).
  destruct (apply_conts is_local G sol noted deps cds keep codes symbols u
              False no_location True Hc Hconsts Hsyms ctx ce ap vars K statics
              extra returns exn Hcount Hvars Hvars_pa Hconts Hconts_pa Hret
              Hexn Hextra)
    as ((Hret_ok & Hfits & Hret_pa) & Hexn_ok & Hexn_pa).
  rewrite Hres in Hret_ok, Hfits, Hret_pa.
  apply known_call_edges_of in Hedges; [ | exact Hnu ].
  destruct Hedges as (witness & apply_node & called_node & Hargs & Hretacc
                      & Hexnacc & Hcalled & Hused & Hcond & Happly).
  destruct (code_facts G noted deps cds keep codes symbols False True Hnd
              statics id c Hcodes Hcode)
    as (_ & cd' & Hcd' & _ & _ & Lp & Lr).
  rewrite Hcd in Hcd'. injection Hcd' as <-.
  destruct (Hcds id cd Hcd) as (Hpar & Hretc & Hexnc & Hcodeid & Hasid).
  set (w := cd_known_arity_call_witness cd) in *.
  pose proof (simple_flows is_local G sol noted cds keep no_location Hc
                Hconsts Hsyms vars _ _ Hvars Hnoted_callee Hcallee)
    as Hfcallee.
  destruct (closure_alloc G sol cds keep no_location statics callee _ id slots
              Hst Hfcallee Hclosure)
    as (b & Hb & Hfb & Hcalled_b).
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & Hcw & _).
  assert (Hw_self : may_come_from sol w w).
  { apply (alloc_self is_local G sol Hc _ _ _ Hcodeid).
    exact (as_hs is_local G sol Hc _ (any_source_input _ _ _ Hc _ Hasid)). }
  assert (Hw : may_come_from sol w witness).
  { apply (may_come_from_alias is_local G sol Hc _ _ apply_node);
      [ exact (cond_alias_alias is_local G sol Hc _ _ _ Hlive Hcond) | ].
    refine (may_come_from_read is_local G sol Hc w b _ _ _ _ _ Hcw Happly Hb
              Hw_self).
    apply (not_local is_local Hlocal); intros; discriminate. }
  assert (Hlive' : live sol (Some id)).
  { simpl.
    refine (read_any_usage is_local G sol Hc w witness called_node _ _ _
              Hcodeid Hcalled Hw (any_usage_input _ _ _ Hc _ Hused)).
    apply (not_local is_local Hlocal); intros; discriminate. }
  apply kinds_eqb_eq in Hk1. apply kinds_per_parameter_eqb_eq in Hk2.
  apply unarize_per_parameter_eq in Hk2. apply arity_eqb_eq in Hk3.
  apply (enter_code_ok is_local G sol noted deps cds keep codes symbols u
           False no_location True Hc Hnd id c callee vs callee_vars marker
           _ statics hp rd Hcode Hlive'); try assumption.
  - rewrite <- (length_map kind_of_value vs), Hk1, Hk2. reflexivity.
  - intros cd' Hcd'. rewrite Hcd in Hcd'. injection Hcd' as <-.
    split; [ | split; [ | split ] ].
    + intros j v p Hv Hp.
      destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
      pose proof (simple_flows is_local G sol noted cds keep no_location Hc
                    Hconsts Hsyms vars _ _ Hvars
                    (Hnoted_args s (nth_error_In _ _ Hs)) Hsv) as Hfs.
      refine (argument_transfer is_local G sol Hc (flows_ v) w witness p
                (node_of_simple s) j _ _ Hw (Hargs j s Hs) (Hpar j p Hp)
                (flows_hs is_local G sol keep _ no_location Hc _ _ Hfs) Hfs).
      * intros x' y'. exact (flows_alias is_local G sol keep _ no_location Hc
                               v x' y').
      * intros x' m'. exact (flows_escape is_local G sol keep _ no_location Hc
                               v x' m').
    + apply (flows_alias is_local G sol keep _ no_location Hc callee _ b
               (alias_propagate _ _ _ Hc _ _ _ Hlive' Hprop)).
      exact Hfb.
    + apply (marker_ok h vars0 K0 rec0 returns (cd_return cd) i x Hret_ok).
      * intros j n' n v _ Hn' Hn Hv.
        refine (flows_read is_local G sol keep _ no_location Hc w witness n n'
                  _ v _ (Hretc j n' Hn') (Hretacc j n Hn) Hw Hv).
        apply (not_local is_local Hlocal); intros; discriminate.
      * destruct Hfits as [Hf | (t & Ht & Hle)].
        -- left. rewrite Lr, <- Hk3, Hf. reflexivity.
        -- right. simpl in Ht. injection Ht as <-. exact Hle.
      * intros n' v Hn' Hv. rewrite Hr in Hn'. injection Hn' as <-.
        rewrite (flows_hs is_local G sol keep _ no_location Hc _ _ Hv) in Hrs.
        discriminate Hrs.
    + apply (cont_ok_compose _ _ _ _ _ _ _ _ _ _ _ _ _ [exn] [cd_exn cd]
               Hexn_ok); [ | left; reflexivity ].
      intros [ | j] n' n Hn' Hn v Hv; [ | destruct j; discriminate Hn' ].
      injection Hn' as <-. injection Hn as <-.
      refine (flows_read is_local G sol keep _ no_location Hc w witness exn
                (cd_exn cd) _ v _ Hexnc Hexnacc Hw Hv).
      apply (not_local is_local Hlocal); intros; discriminate.
  - exact (simples_pa_ok _ _ _ _ Hvars_pa Hvs).
  - exact (simple_pa_ok _ _ _ _ Hvars_pa Hcallee).
  - destruct (cont_pa_closure_inv _ _ _ _ _ Hret_pa) as [Hpa1 Hpa2].
    exact (Closure_pa _ _ _ _ _ Hpa1 Hpa2).
Qed.

(* The run from the call of [marked_call] never binds [x]. *)
Lemma marked_run (s : state) (ap : apply_expr) (callee : value)
    (id : code_id) (slots : list (value_slot * value)) (cd : code_dep)
    (i : nat) (r : node) (h : continuation_handler) (vars0 : env)
    (K0 : cont_env) (rec0 : list (continuation * continuation_handler))
    (x : variable) (m : nat) (s' : state) :
  inv s -> state_expr s = Apply ap ->
  ap_call_kind ap <> Indirect_unknown_arity ->
  ap_exn_continuation ap <> ap_continuation ap ->
  eval_simple (state_vars s) (ap_callee ap) = Some callee ->
  closure_of (state_statics s) callee = Some (id, slots) ->
  lookup id cds = Some cd -> nth_error (cd_return cd) i = Some r ->
  s_has_source sol r = false ->
  resolve (ap_continuation ap) (state_conts s) = Closure h vars0 K0 rec0 ->
  s_has_source sol (Node_variable x) = false ->
  run m
    (State (Apply ap) (state_vars s)
       ((ap_continuation ap, Closure (rename_param i x h) vars0 K0 rec0)
        :: state_conts s)
       (state_statics s) (state_heap s) (state_reads s))
  = Step s' ->
  lookup x (state_vars s') = None.
Proof.
  intros Hs Hexpr Hnu Hexn_ne Hcallee Hclosure Hcd Hr Hrs Hres Hx Hrun.
  destruct m as [ | m].
  - simpl in Hrun. injection Hrun as <-. exact (inv_unbound s x Hs Hx).
  - rewrite run_succ in Hrun.
    pose proof (marked_call s ap callee id slots cd i r h vars0 K0 rec0 x Hs
                  Hexpr Hnu Hexn_ne Hcallee Hclosure Hcd Hr Hrs Hres) as Hs1.
    destruct (step _) as [s1 | | ]; try discriminate Hrun.
    exact (inv_unbound s' x (inv_run m s1 s' Hs1 Hrun) Hx).
Qed.

(* A call of [s] that does not enter the code of its callee with the
   results it expects does not step, whatever its continuations. *)
Lemma unentered_run (s : state) (ap : apply_expr) (K : cont_env)
    (x : variable) (m : nat) (s' : state) :
  inv s -> state_expr s = Apply ap ->
  ap_call_kind ap <> Indirect_unknown_arity ->
  (∀ (callee : value) (id : code_id) (slots : list (value_slot * value))
     (c : code),
     eval_simple (state_vars s) (ap_callee ap) = Some callee ->
     closure_of (state_statics s) callee = Some (id, slots) ->
     lookup id (static_code (state_statics s)) = Some c ->
     arity_eqb (ap_return_arity ap) (code_result_arity c) = true -> False) ->
  s_has_source sol (Node_variable x) = false ->
  run m
    (State (Apply ap) (state_vars s) K (state_statics s) (state_heap s)
       (state_reads s))
  = Step s' ->
  lookup x (state_vars s') = None.
Proof.
  intros Hs Hexpr Hnu Hnot Hx Hrun.
  destruct m as [ | m].
  - simpl in Hrun. injection Hrun as <-. exact (inv_unbound s x Hs Hx).
  - rewrite run_succ in Hrun.
    destruct s as [e0 vars K1 statics hp rd]. simpl in *. subst e0.
    destruct Hs as (ctx & ce & He & _ & _ & _ & _ & _ & Hcodes & _).
    destruct He as (Hfrag & _).
    pose proof (full_call_step G noted deps cds keep codes symbols False True
                  ap vars K statics hp rd Hnu Hfrag Hcodes) as Hfull.
    destruct (step _) as [s1 | | ]; try discriminate Hrun.
    destruct Hfull as (id & callee & vs & extra & slots & c & callee_vars
                       & _ & Hcallee & _ & _ & Hclosure & Hcode & _ & _ & _
                       & _ & _ & Hk3 & _).
    destruct (Hnot callee id slots c Hcallee Hclosure Hcode Hk3).
Qed.

End Frames.

(* ======================================================================== *)
(* The code that a call that never returns enters                           *)
(* ======================================================================== *)

Section Entered.

Variable is_local : field -> bool.
Variable F : facts.
Variable sol : solution.
Variable cds : finite_map code_dep.
Variable codes : finite_map code_metadata.
Hypothesis Hc : closed is_local (graph_of_facts F) sol.
Hypothesis Hlocal : is_local_slots_only is_local.

Abbreviation G := (graph_of_facts F).
Abbreviation kw := (Field_call_witness Known_arity_code_pointer).

(* The call of known arity [ap] reads at [a] the call witness of known
   arity of its callee, which may come from the closure [b] of the code
   [id], with the code dep [cd]. If [apply_never_returns] holds of the
   call, a result of [cd], of a rank that the call expects, has no
   source. *)
Lemma never_returns_result (ap : apply_expr) (a b : node) (id : code_id)
    (cd : code_dep) :
  ap_call_kind ap <> Indirect_unknown_arity ->
  g_accessor G a kw (node_of_simple (ap_callee ap)) ->
  may_come_from sol b (node_of_simple (ap_callee ap)) ->
  s_has_source sol b = true ->
  g_constructor G b kw (cd_known_arity_call_witness cd) ->
  lookup id cds = Some cd -> known_witness_edges G id cd ->
  length (unarize (ap_return_arity ap)) <= length (cd_return cd) ->
  apply_never_returns F sol cds codes ap = true ->
  ∃ (i : nat) (r : node),
    nth_error (cd_return cd) i = Some r ∧ s_has_source sol r = false.
Proof.
  intros Hnu Ha Hb Hhsb Hbw Hcd Hkw Hlen Hnr.
  assert (Hkl : is_local kw = false)
    by (apply (not_local is_local Hlocal); intros; discriminate).
  (* What [apply_never_returns] reads when the code is not known. *)
  assert (Hunknown :
            code_id_actually_called F sol ap = None ->
            never_returns_of_call F sol Known_arity_code_pointer
              (get_direct_sources sol (node_of_simple (ap_callee ap)))
              [length (ap_args ap)] (length (unarize (ap_return_arity ap)))
            = true).
  { intros Ecall. unfold apply_never_returns in Hnr. rewrite Ecall in Hnr.
    destruct (ap_callee ap); [ | | discriminate Hnr ];
      (destruct (ap_call_kind ap);
       [ exact Hnr | exfalso; apply Hnu; reflexivity | exact Hnr ]). }
  set (f := node_of_simple (ap_callee ap)) in *.
  destruct Hkw as (_ & Hretc & _ & Hcodeid & Hasid).
  destruct (alloc_self is_local G sol Hc _ _ _ Hcodeid
              (as_hs is_local G sol Hc _ (any_source_input _ _ _ Hc _ Hasid)))
    as (Hhsw & Hww & _).
  set (w := cd_known_arity_call_witness cd) in *. clearbody w.
  destruct Hb as (_ & Hfb & _).
  destruct (code_id_actually_called F sol ap) as [id0 | ] eqn:Ecall.
  - (* The code is known: it is [id]. *)
    unfold apply_never_returns in Hnr. rewrite Ecall in Hnr.
    destruct (lookup id0 codes) as [md | ]; [ | discriminate Hnr ].
    destruct (FlambdaReaperWrapperProofs.called_some F sol ap id0 Ecall)
      as (ids & Eids & Esing).
    pose proof (FlambdaReaperWrapperProofs.directly_no_any F sol f ids a Eids
                  Ha) as Hasa.
    assert (Hasf : s_any_source sol f = false).
    { destruct (s_any_source sol f) eqn:E; [ | reflexivity ].
      rewrite (any_source_accessor _ _ _ Hc _ _ a E Ha Hkl) in Hasa.
      discriminate. }
    destruct Hfb as [Hfb | Hfb]; [ congruence | ].
    pose proof (alias_accessor_source _ _ _ Hc f b _ a w Ha Hfb Hasf Hbw)
      as Haw.
    assert (Hasw : s_any_source sol w = false).
    { destruct (s_any_source sol w) eqn:E; [ | reflexivity ].
      rewrite (any_source_alias _ _ _ Hc w a Haw E) in Hasa. discriminate. }
    destruct Hww as [Hww | Hww]; [ congruence | ].
    pose proof (sources_alias _ _ _ Hc w a w Hww Hasw Haw Hasa) as Haw'.
    pose proof (FlambdaReaperWrapperProofs.directly_member F sol f ids a w id
                  Eids Ha Haw' Hhsw Hcodeid) as Hin.
    pose proof (FlambdaReaperWrapperProofs.singleton_some ids id0 id Esing
                  Hin) as <-.
    unfold code_never_returns in Hnr. rewrite Hcd in Hnr.
    apply existsb_exists in Hnr as (r & Hr & Hno).
    apply In_nth_error in Hr as (i & Hi).
    exists i, r. split; [ exact Hi | ]. apply negb_true_iff. exact Hno.
  - (* The code is not known: the result is read off [w]. *)
    specialize (Hunknown eq_refl).
    cbn [never_returns_of_call] in Hunknown.
    unfold get_direct_sources in Hunknown.
    destruct (s_any_source sol f) eqn:Hasf; [ discriminate Hunknown | ].
    destruct Hfb as [Hfb | Hfb]; [ congruence | ].
    unfold get_field_sources in Hunknown.
    match type of Hunknown with
    | context [if existsb ?p (f_constructor F) then _ else _] =>
      destruct (existsb p (f_constructor F)) eqn:Eany;
        [ discriminate Hunknown | ]
    end.
    assert (Hasw : s_any_source sol w = false).
    { destruct (s_any_source sol w) eqn:E; [ | reflexivity ].
      rewrite <- Eany. symmetry. apply existsb_exists.
      exists (b, kw, w). split; [ exact Hbw | ].
      rewrite Hfb, Hhsb, field_eqb_refl, E. reflexivity. }
    destruct Hww as [Hww | Hww]; [ congruence | ].
    apply existsb_exists in Hunknown as (i & Hi & Hno).
    apply in_seq in Hi.
    destruct (nth_error (cd_return cd) i) as [r | ] eqn:Hr;
      [ | apply nth_error_None in Hr; lia ].
    exists i, r. split; [ exact Hr | ].
    destruct (s_has_source sol r) eqn:Er; [ | reflexivity ].
    exfalso. apply negb_true_iff, Bool.not_true_iff_false in Hno. apply Hno.
    unfold result_has_source_of. apply existsb_exists.
    exists (w, Field_return_of_call (Normal i), r).
    split; [ exact (Hretc i r Hr) | ].
    rewrite field_eqb_refl, Er, andb_true_r, andb_true_r.
    apply existsb_exists. exists (b, kw, w). split; [ exact Hbw | ].
    rewrite Hfb, Hhsb, field_eqb_refl, Hww, Hhsw. reflexivity.
Qed.

End Entered.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The run of the unit keeps the invariant, with the hypotheses of the
   section [Frames]. *)
Lemma unit_inv (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s : state) :
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop),
    closed is_local (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T)
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e))
        Node_all_constants
    ∧ (∀ s : symbol,
         noted s ->
         may_come_from (solution_of_tables T) (Node_symbol s) (Node_symbol s))
    ∧ (∀ s : symbol,
         among s (static_names symbols_of_binding e) = true ->
         may_come_from (solution_of_tables T) (Node_symbol s) (Node_symbol s))
    ∧ (∀ d : closure_dep,
         deps d ->
         closure_dep_facts (graph_of_facts (traverse_unit flag u e))
           (unit_code_deps flag u e) d)
    ∧ NoDup (map fst (unit_code_deps flag u e))
    ∧ (∀ (id : code_id) (cd : code_dep),
         lookup id (unit_code_deps flag u e) = Some cd ->
         known_witness_edges (graph_of_facts (traverse_unit flag u e)) id cd)
    ∧ FlambdaReaperLocalFieldsSpec.local_reads_used is_local
        (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    ∧ inv (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
        noted deps (unit_code_deps flag u e) (unit_kept e) (unit_codes e)
        (static_names symbols_of_binding e) u s.
Proof.
  intros Hlocal Hslots Hfrag Hcheck Hrun.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  pose proof (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                is_local _ _
                (old_premise_local_reads_used flag is_local u e T Hlocal
                   Hslots))
    as Hlrs.
  destruct (FlambdaReaperRebuildProofs.traverse_unit_keeps flag u e True false
              Hfrag (or_introl I))
    as (noted & deps & cds & He & Hconsts & Himported & _ & Hdeps & Hnd
        & Hkeys & _ & Hcds_eq).
  subst cds.
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  set (cds := unit_code_deps flag u e) in *.
  pose proof (static_names_edges G noted deps cds (unit_kept e) (unit_codes e)
                (static_names symbols_of_binding e) True False) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id' cd' Hl.
    assert (Hid : id' ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst cds (id', cd')).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id' Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd'' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact Hw. }
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s0 Hs. apply among_In in Hs.
    destruct (Hsym_names s0 Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id' | ] eqn:Hl;
        [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id' eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - exact (boxed_self is_local G sol Hc Hconsts _ _ Hb).
    - exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
    - destruct elements as [ | element elements ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))). }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s0 Hs.
    destruct (among s0 (static_names symbols_of_binding e)) eqn:Hunit;
      [ exact (Hunit_syms s0 Hunit) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s0 Hs Hunit)). }
  pose proof (initial_flow_invariant G sol noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e) u False
                no_location True e He) as H0.
  exists noted, deps.
  do 8 (split; [ assumption | ]).
  exact (inv_run is_local G sol noted deps cds (unit_kept e) (unit_codes e)
           (static_names symbols_of_binding e) u Hc Hlocal Hconsts Hsyms
           Hunit_syms Hdeps Hnd Hcds Hlrs n _ _ H0 Hrun).
Qed.

Theorem dead_frame : dead_frame_statement.
Proof.
  intros flag is_local u e T n m s s' ap callee id slots cd i r h vars0 K0
    rec0 x Hlocal Hslots Hfrag Hcheck Hrun Hexpr Hnu Hexn_ne Hcallee
    Hclosure Hcd Hr Hrs Hres Hx Hrun'.
  destruct (unit_inv flag is_local u e T n s Hlocal Hslots Hfrag Hcheck Hrun)
    as (noted & deps & Hc & Hconsts & Hsyms & Hunit_syms & Hdeps & Hnd & Hcds
        & Hlrs & Hs).
  exact (marked_run is_local _ _ noted deps _ _ _ _ u Hc Hlocal Hconsts Hsyms
           Hunit_syms Hdeps Hnd Hcds Hlrs s ap callee id slots cd i r h vars0
           K0 rec0 x m s' Hs Hexpr Hnu Hexn_ne Hcallee Hclosure Hcd Hr Hrs
           Hres Hx Hrun').
Qed.

(* The call of [s] that enters the code [id] of [callee] with the results
   it expects, and that [apply_never_returns] makes one that never
   returns: a result of the code dep of [id] has no source. *)
Lemma entered_dead_result (is_local : field -> bool) (F : facts)
    (sol : solution) (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep) (keep : value_slot -> bool)
    (codes : list (code_id * code)) (symbols : list symbol)
    (u : unit_interface) (md : finite_map code_metadata) (s : state)
    (ap : apply_expr) (callee : value) (id : code_id)
    (slots : list (value_slot * value)) (c : code) :
  closed is_local (graph_of_facts F) sol -> is_local_slots_only is_local ->
  g_any_source (graph_of_facts F) Node_all_constants ->
  (∀ s : symbol, noted s -> may_come_from sol (Node_symbol s) (Node_symbol s))
  -> NoDup (map fst cds) ->
  (∀ (id : code_id) (cd : code_dep),
     lookup id cds = Some cd -> known_witness_edges (graph_of_facts F) id cd)
  -> inv (graph_of_facts F) sol noted deps cds keep codes symbols u s ->
  state_expr s = Apply ap -> ap_call_kind ap <> Indirect_unknown_arity ->
  eval_simple (state_vars s) (ap_callee ap) = Some callee ->
  closure_of (state_statics s) callee = Some (id, slots) ->
  lookup id (static_code (state_statics s)) = Some c ->
  arity_eqb (ap_return_arity ap) (code_result_arity c) = true ->
  apply_never_returns F sol cds md ap = true ->
  ∃ (cd : code_dep) (i : nat) (r : node),
    lookup id cds = Some cd ∧ nth_error (cd_return cd) i = Some r
    ∧ s_has_source sol r = false.
Proof.
  intros Hc Hlocal Hconsts Hsyms Hnd Hcds Hs Hexpr Hnu Hcallee Hclosure Hcode
    Hk3 Hnr.
  set (G := graph_of_facts F) in *.
  destruct s as [e0 vars K statics hp rd]. simpl in *. subst e0.
  destruct Hs as (ctx & ce & He & Hlive & _ & Hvars & _ & Hst & Hcodes & _).
  destruct He as (_ & Hnoted_callee & _ & returns & exn & _ & _ & Hedges).
  apply known_call_edges_of in Hedges; [ | exact Hnu ].
  destruct Hedges as (witness & a & called_node & _ & _ & _ & _ & _ & _
                      & Happly).
  destruct (code_facts G noted deps cds keep codes symbols False True Hnd
              statics id c Hcodes Hcode)
    as (_ & cd & Hcd & _ & _ & _ & Lr).
  pose proof (simple_flows is_local G sol noted cds keep no_location Hc
                Hconsts Hsyms vars _ _ Hvars Hnoted_callee Hcallee)
    as Hfcallee.
  destruct (closure_alloc G sol cds keep no_location statics callee _ id slots
              Hst Hfcallee Hclosure)
    as (b & Hb & Hfb & Hcalled_b).
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (_ & Hcw & _).
  apply arity_eqb_eq in Hk3.
  destruct (never_returns_result is_local F sol cds md Hc Hlocal ap a b id cd
              Hnu Happly Hb
              (flows_hs is_local G sol keep _ no_location Hc _ _ Hfb) Hcw Hcd
              (Hcds id cd Hcd) ltac:(rewrite Lr, Hk3; reflexivity) Hnr)
    as (i & r & Hr & Hrs).
  exists cd, i, r. auto.
Qed.

Theorem dead_frame_of_call : dead_frame_of_call_statement.
Proof.
  intros flag is_local u e T n s s1 ap callee id slots Hlocal Hslots Hfrag
    Hcheck Hrun Hexpr Hnu Hcallee Hclosure Hnr Hstep.
  destruct (unit_inv flag is_local u e T n s Hlocal Hslots Hfrag Hcheck Hrun)
    as (noted & deps & Hc & Hconsts & Hsyms & _ & _ & Hnd & Hcds & _ & Hs).
  assert (Hentered :
            ∃ c, lookup id (static_code (state_statics s)) = Some c
                 ∧ arity_eqb (ap_return_arity ap) (code_result_arity c)
                   = true).
  { destruct s as [e0 vars K statics hp rd]. simpl in *. subst e0.
    destruct Hs as (ctx & ce & He & _ & _ & _ & _ & _ & Hcodes & _).
    destruct He as (Hfrag' & _).
    pose proof (full_call_step _ noted deps _ _ _ _ False True ap vars K
                  statics hp rd Hnu Hfrag' Hcodes) as Hfull.
    rewrite Hstep in Hfull.
    destruct Hfull as (id' & callee' & _ & _ & slots' & c & _ & _ & Hcallee'
                       & _ & _ & Hclosure' & Hcode & _ & _ & _ & _ & _ & Hk3
                       & _).
    rewrite Hcallee in Hcallee'. injection Hcallee' as <-.
    rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
    exists c. auto. }
  destruct Hentered as (c & Hcode & Hk3).
  exact (entered_dead_result is_local _ _ noted deps _ _ _ _ u _ s ap callee
           id slots c Hc Hlocal Hconsts Hsyms Hnd Hcds Hs Hexpr Hnu Hcallee
           Hclosure Hcode Hk3 Hnr).
Qed.

Theorem never_returns_frame : never_returns_frame_statement.
Proof.
  intros flag is_local u e T n s ap h vars0 K0 rec0 Hlocal Hslots Hfrag
    Hcheck Hrun Hexpr Hnu Hexn_ne Hnr Hres.
  destruct (unit_inv flag is_local u e T n s Hlocal Hslots Hfrag Hcheck Hrun)
    as (noted & deps & Hc & Hconsts & Hsyms & Hunit_syms & Hdeps & Hnd & Hcds
        & Hlrs & Hs).
  (* The run from a call that does not enter code never binds [x]. *)
  assert (Hstuck :
            (∀ (callee : value) (id : code_id)
               (slots : list (value_slot * value)) (c : code),
               eval_simple (state_vars s) (ap_callee ap) = Some callee ->
               closure_of (state_statics s) callee = Some (id, slots) ->
               lookup id (static_code (state_statics s)) = Some c ->
               arity_eqb (ap_return_arity ap) (code_result_arity c) = true ->
               False) ->
            ∃ i : nat,
              ∀ (x : variable) (m : nat) (s' : state),
                s_has_source (solution_of_tables T) (Node_variable x) = false
                ->
                run m
                  (State (Apply ap) (state_vars s)
                     ((ap_continuation ap,
                       Closure (rename_param i x h) vars0 K0 rec0)
                      :: state_conts s)
                     (state_statics s) (state_heap s) (state_reads s))
                = Step s' ->
                lookup x (state_vars s') = None).
  { intros Hnot. exists 0%nat. intros x m s' Hx Hrun'.
    exact (unentered_run is_local _ _ noted deps _ _ _ _ u Hc s ap _ x m s'
             Hs Hexpr Hnu Hnot Hx Hrun'). }
  destruct (eval_simple (state_vars s) (ap_callee ap)) as [callee | ]
    eqn:Hcallee; [ | apply Hstuck; intros; discriminate ].
  destruct (closure_of (state_statics s) callee) as [[id slots] | ]
    eqn:Hclosure;
    [ | apply Hstuck; intros callee' ? ? ? E; injection E as <-;
        congruence ].
  destruct (lookup id (static_code (state_statics s))) as [c | ] eqn:Hcode;
    [ | apply Hstuck; intros callee' id' ? ? E1 E2;
        injection E1 as <-; rewrite Hclosure in E2; injection E2 as <- _;
        congruence ].
  destruct (arity_eqb (ap_return_arity ap) (code_result_arity c)) eqn:Hk3;
    [ | apply Hstuck; intros callee' id' ? c' E1 E2 E3;
        injection E1 as <-; rewrite Hclosure in E2; injection E2 as <- _;
        rewrite Hcode in E3; injection E3 as <-; congruence ].
  destruct (entered_dead_result is_local _ _ noted deps _ _ _ _ u _ s ap callee
              id slots c Hc Hlocal Hconsts Hsyms Hnd Hcds Hs Hexpr Hnu Hcallee
              Hclosure Hcode Hk3 Hnr)
    as (cd & i & r & Hcd & Hr & Hrs).
  exists i. intros x m s' Hx Hrun'.
  exact (marked_run is_local _ _ noted deps _ _ _ _ u Hc Hlocal Hconsts Hsyms
           Hunit_syms Hdeps Hnd Hcds Hlrs s ap callee id slots cd i r h vars0
           K0 rec0 x m s' Hs Hexpr Hnu Hexn_ne Hcallee Hclosure Hcd Hr Hrs
           Hres Hx Hrun').
Qed.

(* ======================================================================== *)
(* The return continuation of code whose result has no source               *)
(* ======================================================================== *)

Lemma lookup_rec_conts_other (k : continuation) (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler)) :
  existsb (fun '(k', _) => Pos.eqb k' k) handlers = false ->
  lookup k (FlambdaReaperFlowProofs.rec_conts ce handlers) = lookup k ce.
Proof.
  intros H. unfold FlambdaReaperFlowProofs.rec_conts. rewrite lookup_app.
  induction handlers as [ | [k' h] handlers IH]; [ reflexivity | ].
  cbn [existsb] in H. apply orb_false_iff in H as [Hk H].
  cbn [map]. rewrite lookup_other; [ exact (IH H) | ].
  intros ->. rewrite Pos.eqb_refl in Hk. discriminate Hk.
Qed.

Lemma combine_nth_in {A B : Type} (l : list A) (l' : list B) (i : nat)
    (a : A) (b : B) :
  nth_error l i = Some a -> nth_error l' i = Some b -> (a, b) ∈ combine l l'.
Proof.
  revert l' i. induction l as [ | x l IH]; intros l' i Ha Hb;
    [ destruct i; discriminate Ha | ].
  destruct l' as [ | y l']; [ destruct i; discriminate Hb | ].
  destruct i as [ | i]; cbn in Ha, Hb |- *.
  - injection Ha as <-. injection Hb as <-. left. reflexivity.
  - right. exact (IH l' i Ha Hb).
Qed.

(* What an [Apply_cont] to [k] that [applies_to] finds gives the nodes of
   [k]. *)
Lemma applies_to_edges (G : graph) (noted : symbol -> Prop)
    (deps : closure_dep -> Prop) (cds : finite_map code_dep)
    (keep : value_slot -> bool) (codes : list (code_id * code))
    (symbols : list symbol) (slots_read heap_prims : Prop)
    (k : continuation) (ns : list node) :
  ∀ e : expr, ∀ ctx ce,
    FlambdaReaperFlowProofs.expr_edges G noted deps cds keep codes symbols
      slots_read heap_prims ctx ce e ->
    lookup k ce = Some ns ->
    ∀ ac : apply_cont, ac ∈ applies_to k e ->
      ∀ (p : node) (s : simple),
        (p, s) ∈ combine ns (ac_args ac) -> g_alias G p (node_of_simple s).
Proof.
  assert (H :
    (∀ e : expr, ∀ ctx ce,
       FlambdaReaperFlowProofs.expr_edges G noted deps cds keep codes symbols
         slots_read heap_prims ctx ce e ->
       lookup k ce = Some ns ->
       ∀ ac : apply_cont, ac ∈ applies_to k e ->
         ∀ (p : node) (s : simple),
           (p, s) ∈ combine ns (ac_args ac) -> g_alias G p (node_of_simple s))
    ∧ (∀ h : continuation_handler, ∀ ctx ce,
         FlambdaReaperFlowProofs.handler_edges G noted deps cds keep codes
           symbols slots_read heap_prims ctx ce h ->
         lookup k ce = Some ns ->
         ∀ ac : apply_cont, ac ∈ handler_applies_to k h ->
           ∀ (p : node) (s : simple),
             (p, s) ∈ combine ns (ac_args ac) ->
             g_alias G p (node_of_simple s)));
    [ | exact (proj1 H) ].
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [FlambdaReaperFlowProofs.expr_edges
         FlambdaReaperFlowProofs.handler_edges applies_to handler_applies_to].
  - intros x kind d body IH ctx ce [_ H]. exact (IH ctx ce H).
  - intros bindings body _ IH ctx ce [_ H]. exact (IH ctx ce H).
  - intros k' h body IHh IH ctx ce [Hh H] Hk ac Hac.
    apply in_app_or in Hac as [Hac | Hac];
      [ exact (IHh ctx ce Hh Hk ac Hac) | ].
    destruct (Pos.eqb_spec k' k) as [_ | Hne]; [ destruct Hac | ].
    refine (IH ctx _ H _ ac Hac). rewrite lookup_other; [ exact Hk | ].
    intros ->. apply Hne. reflexivity.
  - intros handlers body IHh IH ctx ce [Hhs H] Hk ac Hac.
    destruct (existsb (fun '(k', _) => Pos.eqb k' k) handlers) eqn:Ex;
      [ destruct Hac | ].
    pose proof (lookup_rec_conts_other k ce handlers Ex) as Hl.
    rewrite <- Hl in Hk.
    apply in_app_or in Hac as [Hac | Hac]; [ | exact (IH ctx _ H Hk ac Hac) ].
    apply in_flat_map in Hac as ([k' h] & Hkh & Hac).
    exact (IHh k' h Hkh ctx _
             (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh) Hk ac Hac).
  - intros ap ctx ce _ _ ac [].
  - intros ac ctx ce [Hal _] Hk ac' Hac.
    destruct (Pos.eqb_spec (ac_continuation ac) k) as [Heq | _];
      [ | destruct Hac ].
    destruct Hac as [<- | []]. apply Hal. rewrite Heq. exact Hk.
  - intros scrutinee arms ctx ce Harms Hk ac Hac.
    apply filter_In in Hac as [Hac Heq]. apply Pos.eqb_eq in Heq.
    apply in_map_iff in Hac as ([d ac'] & <- & Hin).
    destruct (proj1 (FlambdaMapProofs.every_in _ _ _) Harms _ Hin) as [Hal _].
    apply Hal. cbn [snd] in Heq |- *. rewrite Heq. exact Hk.
  - intros ctx ce _ _ ac [].
  - intros params metadata handler IH ctx ce H. exact (IH ctx ce H).
Qed.

Theorem dead_return_dropped : dead_return_dropped_statement.
Proof.
  intros flag is_local u e T id c cd i r ac Hfrag Hcheck Hin Hcd Hr Hrs Hac
    Hlt.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  destruct (FlambdaReaperRebuildProofs.traverse_unit_keeps flag u e True false
              Hfrag (or_introl I))
    as (noted & deps & cds & He & Hconsts & _ & _ & _ & Hnd & _ & _ & ->).
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  pose proof (proj1 (FlambdaReaperWrapperProofs.unit_codes_edges G noted deps
                       (unit_code_deps flag u e) (unit_kept e) (unit_codes e)
                       (static_names symbols_of_binding e) True False)
                e None (unit_conts u) He id c Hin) as Hcode.
  destruct c as [params my_closure ret exn md body].
  cbn [FlambdaReaperFlowProofs.code_edges] in Hcode.
  destruct Hcode as (cd' & Hin' & _ & _ & _ & _ & _ & _ & _ & Hne & _ & _
                     & Hbody).
  rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hcd. injection Hcd as <-.
  cbn [code_return_continuation code_body] in Hac.
  assert (Hk : lookup ret [(exn, [cd_exn cd']); (ret, cd_return cd')]
               = Some (cd_return cd')).
  { rewrite lookup_other; [ apply lookup_head | exact (fun H => Hne H) ]. }
  pose proof (applies_to_edges G noted deps _ _ _ _ True False ret
                (cd_return cd') body _ _ Hbody Hk ac Hac) as Hal.
  destruct (nth_error (ac_args ac) i) as [a | ] eqn:Ha;
    [ | apply nth_error_None in Ha; lia ].
  pose proof (Hal r a (combine_nth_in _ _ i r a Hr Ha)) as Hra.
  assert (Hhs : s_has_source sol (node_of_simple a) = false).
  { destruct (s_has_source sol (node_of_simple a)) eqn:E; [ | reflexivity ].
    rewrite (has_source_alias _ _ _ Hc _ r E (alias_input _ _ _ Hc _ _ Hra))
      in Hrs.
    discriminate Hrs. }
  assert (Hno : simple_has_no_source sol a = true).
  { destruct a as [x | sym | c0]; cbn [node_of_simple simple_has_no_source]
      in Hhs |- *; try (rewrite Hhs; reflexivity).
    rewrite (as_hs is_local G sol Hc _ (any_source_input _ _ _ Hc _ Hconsts))
      in Hhs.
    discriminate Hhs. }
  assert (Hex : existsb (simple_has_no_source sol) (ac_args ac) = true).
  { apply existsb_exists. exists a. split; [ exact (nth_error_In _ _ Ha) | ].
    exact Hno. }
  split; [ exact Hex | split ].
  - intros en. unfold rewrite_apply_cont. rewrite Hex. reflexivity.
  - intros unboxed en. unfold rewrite_apply_cont_u. rewrite Hex. reflexivity.
Qed.

(* An [Apply_cont] to [k] that [applies_to] finds gives [k] as many values
   as [conts_counted] counts. *)
Lemma applies_to_counted (k : continuation) (n : nat) :
  ∀ (e : expr) (ks : finite_map nat),
    conts_counted ks e = true -> lookup k ks = Some n ->
    ∀ ac : apply_cont, ac ∈ applies_to k e -> length (ac_args ac) = n.
Proof.
  assert (H :
    (∀ (e : expr) (ks : finite_map nat),
       conts_counted ks e = true -> lookup k ks = Some n ->
       ∀ ac : apply_cont, ac ∈ applies_to k e -> length (ac_args ac) = n)
    ∧ (∀ (h : continuation_handler) (ks : finite_map nat),
         handler_conts_counted ks h = true -> lookup k ks = Some n ->
         ∀ ac : apply_cont, ac ∈ handler_applies_to k h ->
           length (ac_args ac) = n));
    [ | exact (proj1 H) ].
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [conts_counted handler_conts_counted applies_to handler_applies_to].
  - intros x kind d body IH ks H. exact (IH ks H).
  - intros bindings body _ IH ks H. exact (IH ks H).
  - intros k' h body IHh IH ks H Hk ac Hac.
    apply andb_true_iff in H as [Hh H].
    apply in_app_or in Hac as [Hac | Hac]; [ exact (IHh ks Hh Hk ac Hac) | ].
    destruct (Pos.eqb_spec k' k) as [_ | Hne]; [ destruct Hac | ].
    refine (IH _ H _ ac Hac). rewrite lookup_other; [ exact Hk | ].
    intros ->. apply Hne. reflexivity.
  - intros handlers body IHh IH ks H Hk ac Hac.
    apply andb_true_iff in H as [Hhs H].
    destruct (existsb (fun '(k', _) => Pos.eqb k' k) handlers) eqn:Ex;
      [ destruct Hac | ].
    set (ks' := map (fun '(k0, h0) => (k0, length (handler_params h0)))
                  handlers ++ ks) in *.
    assert (Hk' : lookup k ks' = Some n).
    { unfold ks'. clear - Hk Ex. rewrite lookup_app.
      induction handlers as [ | [k0 h0] handlers IHl]; [ exact Hk | ].
      cbn [existsb] in Ex. apply orb_false_iff in Ex as [Hk0 Ex].
      cbn [map]. rewrite lookup_other; [ exact (IHl Ex) | ].
      intros ->. rewrite Pos.eqb_refl in Hk0. discriminate Hk0. }
    apply in_app_or in Hac as [Hac | Hac]; [ | exact (IH _ H Hk' ac Hac) ].
    apply in_flat_map in Hac as ([k' h] & Hkh & Hac).
    rewrite forallb_forall in Hhs.
    exact (IHh k' h Hkh _ (Hhs _ Hkh) Hk' ac Hac).
  - intros ap ks _ _ ac [].
  - intros ac ks H Hk ac' Hac.
    destruct (Pos.eqb_spec (ac_continuation ac) k) as [Heq | _];
      [ | destruct Hac ].
    destruct Hac as [<- | []]. unfold count_ok in H.
    rewrite Heq, Hk in H. symmetry. apply Nat.eqb_eq. exact H.
  - intros scrutinee arms ks H Hk ac Hac.
    apply filter_In in Hac as [Hac Heq]. apply Pos.eqb_eq in Heq.
    apply in_map_iff in Hac as ([z ac'] & <- & Hin).
    apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
    apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hin.
    rewrite forallb_forall in H. specialize (H _ Hin).
    cbn beta iota in H. unfold count_ok in H. cbn [snd] in Heq.
    rewrite Heq, Hk in H. symmetry. apply Nat.eqb_eq. exact H.
  - intros ks _ _ ac [].
  - intros params metadata handler IH ks H. exact (IH ks H).
Qed.

Theorem dead_returns_not_live : dead_returns_not_live_statement.
Proof.
  intros flag is_local u e T id c ac Hfrag Hcheck Hin Hnr Hac.
  unfold code_never_returns in Hnr.
  destruct (lookup id (unit_code_deps flag u e)) as [cd | ] eqn:Hcd;
    [ | discriminate Hnr ].
  apply existsb_exists in Hnr as (r & Hr & Hrs).
  apply negb_true_iff in Hrs.
  destruct (In_nth_error _ _ Hr) as (i & Hi).
  destruct (FlambdaReaperRebuildProofs.traverse_unit_keeps flag u e True false
              Hfrag (or_introl I))
    as (noted & deps & cds & He & _ & _ & _ & _ & Hnd & _ & _ & ->).
  pose proof (proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                       (graph_of_facts (traverse_unit flag u e)) noted deps
                       (unit_code_deps flag u e) (unit_kept e) (unit_codes e)
                       (static_names symbols_of_binding e) True False)
                e None (unit_conts u) He id c Hin) as Hcodes.
  destruct c as [params my_closure ret exn md body].
  cbn [FlambdaReaperFlowProofs.code_edges] in Hcodes.
  destruct Hcodes as (cd' & Hin' & _ & _ & _ & _ & Hlen & _ & _ & Hne & _
                      & Hcc & _).
  pose proof (lookup_nodup_in _ _ _ Hnd Hin') as Hcd'.
  rewrite Hcd' in Hcd. injection Hcd as <-.
  cbn [code_return_continuation code_body] in Hac |- *.
  assert (Hk : lookup ret
                 [(exn, 1); (ret, length (unarize (cm_result_arity md)))]
               = Some (length (unarize (cm_result_arity md)))).
  { rewrite lookup_other; [ apply lookup_head | exact (fun H => Hne H) ]. }
  pose proof (applies_to_counted ret _ body _ Hcc Hk ac Hac) as Hargs.
  assert (Hlt : i < length (ac_args ac)).
  { rewrite Hargs, <- Hlen. apply nth_error_Some. rewrite Hi.
    discriminate. }
  destruct (dead_return_dropped flag is_local u e T id
              (Code params my_closure ret exn md body) cd' i r ac Hfrag
              Hcheck Hin Hcd' Hi Hrs Hac Hlt)
    as [Hex _].
  unfold FlambdaReaperUnboxingRefinesCallsSpec.live_apply_cont.
  rewrite Hex. reflexivity.
Qed.

Theorem unused_never_returns_example :
  unused_never_returns_example_statement.
Proof.
  cbv zeta. split.
  - apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity.
  - repeat split; vm_compute; reflexivity.
Qed.

(* ======================================================================== *)
(* Calls of unknown arity                                                   *)
(* ======================================================================== *)

(* The list [l] with its element of rank [i] replaced by [a]; [l] if it
   has none. *)
Fixpoint replace_nth {A : Type} (i : nat) (a : A) (l : list A) : list A :=
  match l, i with
  | [], _ => []
  | _ :: l, O => a :: l
  | b :: l, S i => b :: replace_nth i a l
  end.

Lemma replace_nth_other {A : Type} (i j : nat) (a : A) (l : list A) :
  j <> i -> nth_error (replace_nth i a l) j = nth_error l j.
Proof.
  revert i j. induction l as [ | b l IH]; intros i j Hji;
    [ destruct i, j; reflexivity | ].
  destruct i as [ | i], j as [ | j]; simpl;
    [ contradiction | reflexivity | reflexivity | ].
  apply IH. lia.
Qed.

Lemma replace_nth_at {A : Type} (i : nat) (a b : A) (l : list A) :
  nth_error (replace_nth i a l) i = Some b -> b = a.
Proof.
  revert i. induction l as [ | c l IH]; intros i H;
    [ destruct i; discriminate H | ].
  destruct i as [ | i]; simpl in H; [ congruence | exact (IH i H) ].
Qed.

Lemma replace_nth_length {A : Type} (i : nat) (a : A) (l : list A) :
  length (replace_nth i a l) = length l.
Proof.
  revert i. induction l as [ | c l IH]; intros i; [ destruct i; reflexivity | ].
  destruct i as [ | i]; simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Section Unknown.

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
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
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
Hypothesis Hlrs : FlambdaReaperLocalFieldsSpec.local_reads_used is_local G sol.

Local Abbreviation inv_ := (inv G sol noted deps cds keep codes symbols u).
Local Abbreviation flows_ := (flows G sol keep (called_by G cds) no_location).
Local Abbreviation pending_ :=
  (FlambdaReaperFlowProofs.pending G sol cds keep no_location).

(* The results of rank [i] of what the last witness of the call [ap] of
   unknown arity stands for have no source, whatever chain of witnesses
   the graph gives the call. *)
Definition dead_rank (ap : apply_expr) (i : nat) : Prop :=
  ∀ (ws returns : list node) (exn called w W R : node),
    length ws = length (call_groups ap) ->
    caller_chain_edges G returns exn called (combine (call_groups ap) ws) ->
    nth_error ws (pred (length ws)) = Some w ->
    may_come_from sol W w ->
    g_constructor G W (Field_return_of_call (Normal i)) R ->
    s_has_source sol R = false.

(* The same, of the chains of witnesses that the call reads off its
   callee: the first is an alias of the read of the call witness of
   unknown arity off the callee. *)
Definition call_dead_rank (ap : apply_expr) (i : nat) : Prop :=
  ∀ (ctx : option code_id) (ws returns : list node)
    (exn called apply_node w W R : node),
    length ws = length (call_groups ap) ->
    caller_chain_edges G returns exn called (combine (call_groups ap) ws) ->
    (∀ w0 : node, hd_error ws = Some w0 -> cond_alias_edge G ctx w0 apply_node)
    ->
    g_accessor G apply_node (Field_call_witness Unknown_arity_code_pointer)
      (node_of_simple (ap_callee ap)) ->
    nth_error ws (pred (length ws)) = Some w ->
    may_come_from sol W w ->
    g_constructor G W (Field_return_of_call (Normal i)) R ->
    s_has_source sol R = false.

Lemma dead_rank_call (ap : apply_expr) (i : nat) :
  dead_rank ap i -> call_dead_rank ap i.
Proof.
  intros Hdead ctx ws returns exn called apply_node w W R Hlen Hch _ _.
  exact (Hdead ws returns exn called w W R Hlen Hch).
Qed.

(* A call whose last witness gives no source at rank [i] may return to
   nodes whose rank [i] is a variable with no source. *)
Lemma pending_marked (start : node) (ws : list node) (vs : list value)
    (a : arity) (returns : list node) (exn : node) (i : nat) (x : variable) :
  (∀ (w W R : node),
     nth_error ws (pred (length ws)) = Some w ->
     may_come_from sol W w ->
     g_constructor G W (Field_return_of_call (Normal i)) R ->
     s_has_source sol R = false) ->
  pending_ start ws vs a returns exn ->
  pending_ start ws vs a (replace_nth i (Node_variable x) returns) exn.
Proof.
  intros Hdead (Hl & Hch & Hargs & Hlast & Hexn & Hused).
  split; [ exact Hl | split; [ exact Hch | split; [ exact Hargs | ] ] ].
  split; [ | split; [ exact Hexn | exact Hused ] ].
  intros w Hw. destruct (Hlast w Hw) as [Hres H0]. split.
  - intros j r W R v Hr HW HR Hv.
    destruct (Nat.eq_dec j i) as [-> | Hji].
    + exfalso.
      pose proof (flows_hs is_local G sol keep _ no_location Hc _ _ Hv) as H.
      rewrite (Hdead w W R Hw HW HR) in H. discriminate H.
    + rewrite replace_nth_other in Hr by exact Hji.
      exact (Hres j r W R v Hr HW HR Hv).
  - intros r y Hr Hy.
    destruct (Nat.eq_dec 0 i) as [<- | H0i].
    + exfalso. destruct Hy as (W & R & HW & HR & (Hhs & _)).
      rewrite (Hdead w W R Hw HW HR) in Hhs. discriminate Hhs.
    + rewrite replace_nth_other in Hr by (intros E; apply H0i; exact E).
      exact (H0 r y Hr Hy).
Qed.


(* The call of unknown arity [ap] of [s], made with its continuation the
   handler it returns to with its parameter of rank [i] renamed: if the
   results of rank [i] of its last witness have no source, the step keeps
   the invariant. *)
Lemma marked_indirect_call_at (s : state) (ap : apply_expr) (i : nat)
    (h : continuation_handler) (vars0 : env) (K0 : cont_env)
    (rec0 : list (continuation * continuation_handler)) (x : variable) :
  inv_ s -> state_expr s = Apply ap ->
  ap_call_kind ap = Indirect_unknown_arity ->
  ap_exn_continuation ap <> ap_continuation ap ->
  call_dead_rank ap i ->
  resolve (ap_continuation ap) (state_conts s) = Closure h vars0 K0 rec0 ->
  s_has_source sol (Node_variable x) = false ->
  match
    step
      (State (Apply ap) (state_vars s)
         ((ap_continuation ap, Closure (rename_param i x h) vars0 K0 rec0)
          :: state_conts s)
         (state_statics s) (state_heap s) (state_reads s))
  with
  | Step s1 => inv_ s1
  | _ => True
  end.
Proof.
  intros Hs Hexpr Hkind Hexn_ne Hdead Hres Hx.
  destruct s as [e0 vars K statics hp rd]. simpl in *. subst e0.
  destruct Hs as (ctx & ce & He & Hlive & Hcount & Hvars & Hconts & Hst
                  & Hcodes & Hpa).
  destruct Hpa as (Hvars_pa & Hspa & Hconts_pa). simpl in *.
  destruct He as (Hfrag & Hnoted_callee & Hnoted_args & returns & exn & Hret
                  & Hexn & Hedges).
  unfold call_edges in Hedges. rewrite Hkind in Hedges.
  destruct Hedges as (ws & apply_node & called_node & Hlen & Hch & Hused
                      & Hcond & Happly).
  set (marker := Closure (rename_param i x h) vars0 K0 rec0).
  set (returns' := replace_nth i (Node_variable x) returns).
  match goal with
  | |- match ?r with Step _ => _ | _ => _ end =>
    enough (Hr : result_ok G sol noted deps cds keep codes symbols u False
                   no_location True r)
  end.
  { match goal with
    | |- match ?r with Step _ => _ | _ => _ end => destruct r
    end; [ exact Hr | exact I | exact I ]. }
  rewrite step_apply.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs; [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ] eqn:Hextra;
    [ | exact I ].
  destruct (apply_conts is_local G sol noted deps cds keep codes symbols u
              False no_location True Hc Hconsts Hsyms ctx ce ap vars K statics
              extra returns exn Hcount Hvars Hvars_pa Hconts Hconts_pa Hret
              Hexn Hextra)
    as ((Hret_ok & Hret_fits & Hret_pa) & Hexn_ok & Hexn_pa).
  rewrite resolve_head, (resolve_other _ _ _ _ Hexn_ne), Hkind.
  rewrite Hres in Hret_ok, Hret_fits, Hret_pa.
  assert (Hind : FlambdaReaperFlowProofs.indirect_ok codes).
  { intros id c Hin. unfold apply_in_flow_fragment in Hfrag.
    rewrite Hkind in Hfrag. rewrite forallb_forall in Hfrag.
    specialize (Hfrag (id, c) Hin). simpl in Hfrag.
    destruct (code_is_tupled c); [ exact Hfrag | ].
    apply negb_true_iff in Hfrag.
    intros H; rewrite H in Hfrag; discriminate Hfrag. }
  assert (Hlen' : length returns' = length returns)
    by apply replace_nth_length.
  assert (Hmok : cont_ok G sol noted deps cds keep codes symbols u False
                   no_location True marker returns').
  { apply (marker_ok G sol noted deps cds keep codes symbols u h vars0 K0
             rec0 returns returns' i x Hret_ok).
    - intros j n' n v Hji Hn' Hn Hv. unfold returns' in Hn'.
      rewrite (replace_nth_other _ _ _ _ Hji), Hn in Hn'.
      injection Hn' as <-. exact Hv.
    - left. exact Hlen'.
    - intros n' v Hn' Hv. unfold returns' in Hn'.
      apply replace_nth_at in Hn'. subst n'.
      rewrite (flows_hs is_local G sol keep _ no_location Hc _ _ Hv) in Hx.
      discriminate Hx. }
  assert (Hmfits : ∀ n : nat,
             FlambdaReaperFlowProofs.fits u
               (Closure h vars0 K0 rec0) returns n ->
             FlambdaReaperFlowProofs.fits u marker returns' n).
  { intros n [Hf | (t & Ht & Hle)]; [ left; rewrite Hlen'; exact Hf | ].
    right. exists t. split; [ | rewrite Hlen'; exact Hle ].
    simpl in Ht |- *. destruct h as [params md body].
    unfold handler_params in Ht |- *. simpl in Ht |- *.
    rewrite length_map, rename_nth_length. rewrite length_map in Ht.
    exact Ht. }
  assert (Hmpa : cont_pa_ok statics marker).
  { destruct (cont_pa_closure_inv _ _ _ _ _ Hret_pa) as [Hpa1 Hpa2].
    exact (Closure_pa _ _ _ _ _ Hpa1 Hpa2). }
  apply (call_ok is_local G sol noted deps cds keep codes symbols u False
           no_location True Hc Hlocal Hnd Hcds callee
           (node_of_simple (ap_callee ap)) vs (ap_args_arity ap)
           (ap_return_arity ap) ws returns' exn marker _ statics hp rd Hind);
    try assumption.
  - exact (simple_flows is_local G sol noted cds keep no_location Hc Hconsts
             Hsyms vars _ _ Hvars Hnoted_callee Hcallee).
  - exact (simple_pa_ok _ _ _ _ Hvars_pa Hcallee).
  - intros Hne. apply pending_marked.
    + intros w W R Hw HW HR.
      exact (Hdead ctx ws returns exn called_node apply_node w W R Hlen Hch
               Hcond Happly Hw HW HR).
    + exact (indirect_pending is_local G sol noted cds keep no_location Hc
               Hlocal Hconsts Hsyms ctx ap vars vs returns exn ws apply_node
               called_node Hlive Hvars Hnoted_args Hvs Hne Hlen Hch Hused
               Hcond Happly).
  - exact (simples_pa_ok _ _ _ _ Hvars_pa Hvs).
  - exact (Hmfits _ Hret_fits).
  - intros v Hv Hv_pa Hone.
    apply (apply_ok is_local G sol noted deps cds keep codes symbols u False
             no_location True Hc Hlocal Hnd Hcds marker returns' [v] statics
             hp rd Hmok Hmpa);
      try assumption.
    + intros [ | j] y r Hy Hr; [ injection Hy as <-; exact (Hv r Hr)
                               | destruct j; discriminate Hy ].
    + split; [ exact Hv_pa | exact I ].
    + apply Hmfits. destruct Hret_fits as [Hf | Hf]; [ left | right; exact Hf ].
      rewrite <- Hf, Hone. reflexivity.
Qed.

Lemma marked_indirect_call (s : state) (ap : apply_expr) (i : nat)
    (h : continuation_handler) (vars0 : env) (K0 : cont_env)
    (rec0 : list (continuation * continuation_handler)) (x : variable) :
  inv_ s -> state_expr s = Apply ap ->
  ap_call_kind ap = Indirect_unknown_arity ->
  ap_exn_continuation ap <> ap_continuation ap ->
  dead_rank ap i ->
  resolve (ap_continuation ap) (state_conts s) = Closure h vars0 K0 rec0 ->
  s_has_source sol (Node_variable x) = false ->
  match
    step
      (State (Apply ap) (state_vars s)
         ((ap_continuation ap, Closure (rename_param i x h) vars0 K0 rec0)
          :: state_conts s)
         (state_statics s) (state_heap s) (state_reads s))
  with
  | Step s1 => inv_ s1
  | _ => True
  end.
Proof.
  intros Hs Hexpr Hkind Hexn_ne Hdead.
  exact (marked_indirect_call_at s ap i h vars0 K0 rec0 x Hs Hexpr Hkind
           Hexn_ne (dead_rank_call ap i Hdead)).
Qed.

(* The run from the call of [marked_indirect_call_at] never binds [x]. *)
Lemma marked_indirect_run_at (s : state) (ap : apply_expr) (i : nat)
    (h : continuation_handler) (vars0 : env) (K0 : cont_env)
    (rec0 : list (continuation * continuation_handler)) (x : variable)
    (m : nat) (s' : state) :
  inv_ s -> state_expr s = Apply ap ->
  ap_call_kind ap = Indirect_unknown_arity ->
  ap_exn_continuation ap <> ap_continuation ap ->
  call_dead_rank ap i ->
  resolve (ap_continuation ap) (state_conts s) = Closure h vars0 K0 rec0 ->
  s_has_source sol (Node_variable x) = false ->
  run m
    (State (Apply ap) (state_vars s)
       ((ap_continuation ap, Closure (rename_param i x h) vars0 K0 rec0)
        :: state_conts s)
       (state_statics s) (state_heap s) (state_reads s))
  = Step s' ->
  lookup x (state_vars s') = None.
Proof.
  intros Hs Hexpr Hkind Hexn_ne Hdead Hres Hx Hrun.
  destruct m as [ | m].
  - simpl in Hrun. injection Hrun as <-.
    exact (inv_unbound is_local G sol noted deps cds keep codes symbols u Hc
             s x Hs Hx).
  - rewrite run_succ in Hrun.
    pose proof (marked_indirect_call_at s ap i h vars0 K0 rec0 x Hs Hexpr
                  Hkind Hexn_ne Hdead Hres Hx) as Hs1.
    destruct (step _) as [s1 | | ]; try discriminate Hrun.
    exact (inv_unbound is_local G sol noted deps cds keep codes symbols u Hc
             s' x
             (inv_run is_local G sol noted deps cds keep codes symbols u Hc
                Hlocal Hconsts Hsyms Hunit_syms Hdeps Hnd Hcds Hlrs m s1 s'
                Hs1 Hrun)
             Hx).
Qed.

(* The run from the call of [marked_indirect_call] never binds [x]. *)
Lemma marked_indirect_run (s : state) (ap : apply_expr) (i : nat)
    (h : continuation_handler) (vars0 : env) (K0 : cont_env)
    (rec0 : list (continuation * continuation_handler)) (x : variable)
    (m : nat) (s' : state) :
  inv_ s -> state_expr s = Apply ap ->
  ap_call_kind ap = Indirect_unknown_arity ->
  ap_exn_continuation ap <> ap_continuation ap ->
  dead_rank ap i ->
  resolve (ap_continuation ap) (state_conts s) = Closure h vars0 K0 rec0 ->
  s_has_source sol (Node_variable x) = false ->
  run m
    (State (Apply ap) (state_vars s)
       ((ap_continuation ap, Closure (rename_param i x h) vars0 K0 rec0)
        :: state_conts s)
       (state_statics s) (state_heap s) (state_reads s))
  = Step s' ->
  lookup x (state_vars s') = None.
Proof.
  intros Hs Hexpr Hkind Hexn_ne Hdead.
  exact (marked_indirect_run_at s ap i h vars0 K0 rec0 x m s' Hs Hexpr Hkind
           Hexn_ne (dead_rank_call ap i Hdead)).
Qed.

End Unknown.
