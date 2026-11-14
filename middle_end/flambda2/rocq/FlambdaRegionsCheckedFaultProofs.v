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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   a state that has the property stops without a step only where [step] has
   none. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedReadsProofs
  FlambdaRegionsCheckedViewsProofs.
Import ListNotations.

#[local] Transparent eval_prim.

(* Where the plain step has none, the fault is that. *)
Lemma plain_step_fault (rs : region_state) (fault : region_fault) :
  plain_step rs = Region_undefined fault -> fault = No_plain_step.
Proof.
  unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)));
    intros [= <-]; reflexivity.
Qed.

Lemma static_step_fault (rs : region_state) (fault : region_fault) :
  static_step rs = Region_undefined fault -> fault = No_plain_step.
Proof.
  unfold static_step. destruct (step (rs_state rs)); intros [= <-];
    reflexivity.
Qed.

Lemma region_apply_cont_fault (rs : region_state) (ac : apply_cont)
    (fault : region_fault) :
  region_apply_cont rs ac = Region_undefined fault -> fault = No_plain_step.
Proof.
  unfold region_apply_cont. cbv zeta.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try exact (plain_step_fault rs fault).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ];
    [ | intros [= <-]; reflexivity ].
  destruct (kinds_eqb _ _); [ | intros [= <-]; reflexivity ].
  destruct (bind_params params vs def_vars);
    [ discriminate | intros [= <-]; reflexivity ].
Qed.

(* A mark that is on the stack can be cut at. *)
Lemma on_stack_cut (m : mark) (frames : list frame) :
  on_stack m frames = true ->
  ∃ above below : list frame, cut m frames = Some (above, below).
Proof.
  induction frames as [ | f frames IH]; [ discriminate | ].
  cbn [on_stack existsb cut].
  destruct (Pos.eqb (frame_mark f) m);
    [ intros _; eexists _, _; reflexivity | ].
  cbn [orb]. intros H. destruct (IH H) as (above & below & ->).
  eexists _, _. reflexivity.
Qed.

(* The symbols of the function slots of a set. *)
Lemma number_slots_lookup (function_decls : list (function_slot * code_id)) :
  ∀ (next : symbol) (slot : function_slot),
    is_some (lookup slot function_decls) = true ->
    is_some (lookup slot (number_slots function_decls next)) = true.
Proof.
  induction function_decls as [ | [slot0 code] function_decls IH];
    intros next slot; [ discriminate | ].
  cbn [lookup number_slots]. destruct (Pos.eqb slot slot0); [ reflexivity | ].
  apply IH.
Qed.

(* What an allocation makes is what a local allocation binds. *)
Lemma allocation_localises (statics : static_env) (vars : env) (d : named)
    (v : value) (next : symbol) :
  is_allocation d = true ->
  eval_named statics vars d = Some v ->
  ∃ (sym : symbol) (bindings : list (symbol * statically_allocated_value)),
    localise next v = Some (sym, bindings).
Proof.
  destruct d as [s | [u | b | p | hp | rp] args | code slots
                | slot decls slots | | | ];
    try discriminate; cbn [is_allocation eval_named].
  - destruct u; try discriminate. intros _.
    destruct (eval_simples vars args) as [[ | w [ | w' vs]] | ]; cbn;
      try discriminate.
    destruct w; try discriminate.
    unfold eval_prim, eval_unary_primitive.
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros [= <-]. eexists _, _. reflexivity.
  - intros _. destruct (eval_simples vars args) as [vs | ]; cbn;
      [ | discriminate ].
    unfold eval_prim, eval_variadic_primitive.
    destruct p as [[tag] | kind]; cbn.
    + destruct (_ && _); [ | discriminate ]. intros [= <-].
      eexists _, _. reflexivity.
    + destruct (array_made kind vs) as [elements | ]; cbn; [ | discriminate ].
      destruct (forallb _ vs); [ | discriminate ]. intros [= <-].
      eexists _, _. reflexivity.
  - intros _. destruct (eval_value_slots vars slots) as [vs | ]; cbn;
      [ | discriminate ].
    intros [= <-]. eexists _, _. reflexivity.
  - intros _. destruct (lookup slot decls) as [code | ] eqn:Hslot; cbn;
      [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ]; cbn; [ | discriminate ].
    intros [= <-]. cbn [localise].
    pose proof (number_slots_lookup decls next slot) as Hsome.
    rewrite Hslot in Hsome. specialize (Hsome eq_refl).
    destruct (lookup slot (number_slots decls next)) as [s | ];
      [ | discriminate ].
    cbn. eexists _, _. reflexivity.
Qed.

(* No simple that the check takes to live for ever is local (decision
   D46). *)
Lemma forever_none_local (rs : region_state) (cs : check_state)
    (ms : list mark) (simples : list simple) :
  symbols_fresh rs = true ->
  running_checked rs cs ms ->
  (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
  forallb (fun s => is_forever (lifetime_of cs s)) simples = true ->
  existsb (simple_local rs) simples = false.
Proof.
  intros Hfresh Hrun Hbelow Hforever.
  destruct (existsb (simple_local rs) simples) eqn:Hex; [ | reflexivity ].
  apply existsb_exists in Hex as (s & Hin & Hlocal).
  unfold simple_local in Hlocal.
  destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ] eqn:Hv;
    [ | discriminate ].
  pose proof
    (running_forever_heap rs cs ms s v Hrun (Hbelow s Hin)
       (forallb_in _ _ s Hforever Hin) Hv)
    as Hheap.
  rewrite (heap_value_not_local rs v Hfresh Hheap) in Hlocal. discriminate.
Qed.

Lemma checked_state_fault : checked_fault_statement.
Proof.
  intros cert codes rs fault
    (Hinv & Hord & Hcode & Hlocal & Hframes & Hstatics & Hplain & Hpaps & cs
     & ms & Hcheck & Hincl & Hrun & Hconts)
    Hstep.
  pose proof (invariant_fresh rs Hinv) as Hfresh.
  pose proof
    (simples_ok_no_fault rs _ _
       (head_simples_ok cert codes rs cs ms Hfresh Hcheck Hrun Hconts))
    as Hnofault.
  assert (Hbelow :
    ∀ s : simple,
      In s (evaluated_simples (state_expr (rs_state rs))) ->
      simple_below (rs_base rs) s = true)
    by (intros s Hin; exact (head_simples_below rs s Hfresh Hin)).
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He.
  - destruct (region_named d) eqn:Hd.
    + (* a primitive of regions *)
      destruct d as [s | [u | b | v | hp | p] args | | | | | ];
        try discriminate.
      rewrite (region_step_region_let rs x kind p args body He) in Hstep.
      rewrite check_expr_region_let in Hcheck.
      unfold region_let_step in Hstep. unfold check_region_let in Hcheck.
      destruct p, args as [ | [r | sym | c] [ | arg' args]];
        try discriminate;
        try (apply andb_true_iff in Hcheck as [Hcheck _];
             apply andb_true_iff in Hcheck as [Hkind _];
             unfold begin_region in Hstep; rewrite Hkind in Hstep;
             discriminate).
      all: apply andb_true_iff in Hcheck as [Hcheck _];
        apply andb_true_iff in Hcheck as [Hcheck Hopen];
        apply andb_true_iff in Hcheck as [Hkind _];
        destruct (running_region_open rs cs ms r Hrun Hopen)
          as (m & Hr & Hon);
        destruct (on_stack_cut m _ Hon) as (above & below & Hcut);
        unfold end_region in Hstep; cbv zeta in Hstep;
        rewrite Hkind in Hstep;
        change (state_vars (rs_state (count_uses rs)))
          with (state_vars (rs_state rs)) in Hstep;
        change (rs_frames (count_uses rs)) with (rs_frames rs) in Hstep;
        cbn [eval_simple] in Hstep; rewrite Hr, Hcut in Hstep;
        discriminate.
    + (* another [Let] *)
      rewrite (region_step_let rs x kind d body He Hd) in Hstep
        by (rewrite He; exact Hnofault).
      destruct (named_needs_heap (state_vars (rs_state rs)) d);
        [ discriminate | ].
      rewrite (check_expr_let cert codes cs x kind d body Hd) in Hcheck.
      unfold let_step in Hstep. unfold check_let in Hcheck. cbv zeta in Hcheck.
      apply andb_true_iff in Hcheck as [Hcheck Hrest].
      apply andb_true_iff in Hcheck as [_ Husable].
      assert (Hsimples : evaluated_simples (Let x kind d body)
                         = simples_of_named d).
      { destruct d as [s | [u | b | v | hp | p] args | | | | | ];
          first [ reflexivity | discriminate ]. }
      rewrite Hsimples in Hbelow.
      destruct (marker_of x body) as [ | y region body' | ].
      * destruct (is_allocation d) eqn:Halloc; cbn [andb] in Hstep.
        -- apply andb_true_iff in Hrest as [Hforever _].
           change (existsb (simple_local (count_uses rs)) (simples_of_named d))
             with (existsb (simple_local rs) (simples_of_named d)) in Hstep.
           rewrite (forever_none_local rs cs ms _ Hfresh Hrun Hbelow Hforever)
             in Hstep.
           exact (plain_step_fault _ _ Hstep).
        -- exact (plain_step_fault _ _ Hstep).
      * apply andb_true_iff in Hrest as [Hrest _].
        apply andb_true_iff in Hrest as [Hrest _].
        apply andb_true_iff in Hrest as [Hrest Hregion].
        apply andb_true_iff in Hrest as [Hrest _].
        apply andb_true_iff in Hrest as [Halloc Hkind].
        unfold local_allocation in Hstep. cbv zeta in Hstep.
        rewrite Halloc, Hkind in Hstep. cbn [andb] in Hstep.
        assert (Hnamed : ∃ other : bool,
                  named_region (count_uses rs) region = Some other).
        { unfold named_region. destruct region as [[r | sym | c] | ];
            try discriminate; [ | eexists; reflexivity ].
          destruct (running_region_open rs cs ms r Hrun Hregion)
            as (m & Hr & Hon).
          change (state_vars (rs_state (count_uses rs)))
            with (state_vars (rs_state rs)).
          change (rs_frames (count_uses rs)) with (rs_frames rs).
          cbn [eval_simple]. rewrite Hr, Hon.
          destruct (newest_plain (rs_frames rs)); eexists; reflexivity. }
        destruct Hnamed as (other & Hnamed). rewrite Hnamed in Hstep.
        destruct (eval_named (live_statics (count_uses rs))
                    (state_vars (rs_state (count_uses rs))) d)
          as [v | ] eqn:Hv; [ | injection Hstep as <-; reflexivity ].
        destruct (allocation_localises _ _ d v (rs_next (count_uses rs)) Halloc
                    Hv)
          as (sym & bindings & Hlocalise).
        rewrite Hlocalise in Hstep.
        destruct (rs_frames (count_uses rs)) as [ | [m mark_kind locals] ?];
          discriminate.
      * discriminate.
  - (* a [Let_static] *)
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    cbn [check_expr] in Hcheck.
    apply andb_true_iff in Hcheck as [Hcheck _].
    apply andb_true_iff in Hcheck as [Hforever _].
    change (existsb (simple_local (count_uses rs))
              (evaluated_simples (Let_static bindings body)))
      with (existsb (simple_local rs)
              (flat_map simples_of_binding bindings)) in Hstep.
    cbn [evaluated_simples] in Hbelow.
    rewrite (forever_none_local rs cs ms _ Hfresh Hrun Hbelow Hforever)
      in Hstep.
    exact (static_step_fault _ _ Hstep).
  - unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    exact (plain_step_fault _ _ Hstep).
  - unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    exact (plain_step_fault _ _ Hstep).
  - unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    exact (plain_step_fault _ _ Hstep).
  - unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (existsb (simple_open_mark (count_uses rs)) (ac_args ac));
      [ exact (region_apply_cont_fault _ _ _ Hstep)
      | exact (plain_step_fault _ _ Hstep) ].
  - unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    exact (plain_step_fault _ _ Hstep).
  - unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    exact (plain_step_fault _ _ Hstep).
Qed.
