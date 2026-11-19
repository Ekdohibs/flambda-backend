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
   run from there, so [x] is never bound. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaReaperCallingConventions FlambdaReaperDeadFramesSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperLocalFieldsSpec FlambdaReaperLocalFieldsProofs.
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

End Frames.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem dead_frame : dead_frame_statement.
Proof.
  intros flag is_local u e T n m s s' ap callee id slots cd i r h vars0 K0
    rec0 x Hlocal Hslots Hfrag Hcheck Hrun Hexpr Hnu Hexn_ne Hcallee
    Hclosure Hcd Hr Hrs Hres Hx Hrun'.
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
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  rewrite <- Hcds_eq in Hcd.
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
  pose proof (inv_run is_local G sol noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e) u Hc Hlocal
                Hconsts Hsyms Hunit_syms Hdeps Hnd Hcds Hlrs n _ _ H0 Hrun)
    as Hs.
  destruct m as [ | m].
  - simpl in Hrun'. injection Hrun' as <-. simpl.
    exact (inv_unbound is_local G sol noted deps cds (unit_kept e)
             (unit_codes e) (static_names symbols_of_binding e) u Hc s x Hs
             Hx).
  - rewrite run_succ in Hrun'.
    pose proof (marked_call is_local G sol noted deps cds (unit_kept e)
                  (unit_codes e) (static_names symbols_of_binding e) u Hc
                  Hlocal Hconsts Hsyms Hnd Hcds s ap callee id slots cd i r h
                  vars0 K0 rec0 x Hs Hexpr Hnu Hexn_ne Hcallee Hclosure Hcd Hr
                  Hrs Hres)
      as Hs1.
    destruct (step _) as [s1 | | ]; try discriminate Hrun'.
    exact (inv_unbound is_local G sol noted deps cds (unit_kept e)
             (unit_codes e) (static_names symbols_of_binding e) u Hc s' x
             (inv_run is_local G sol noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e) u Hc
                Hlocal Hconsts Hsyms Hunit_syms Hdeps Hnd Hcds Hlrs m _ _ Hs1
                Hrun')
             Hx).
Qed.
