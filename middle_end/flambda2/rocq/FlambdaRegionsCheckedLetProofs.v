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
   the value that a [Let] binds is as the lifetime that the check gives it
   says, and is good. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaClosuresProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsReadsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs FlambdaRegionsCheckedCallProofs
  FlambdaRegionsCheckedAppliedProofs FlambdaRegionsCheckedMovesProofs
  FlambdaRegionsCheckedDeepProofs.
Import ListNotations.

(* ======================================================================== *)
(* Primitives that give a number                                            *)
(* ======================================================================== *)

Lemma tagged_const (v : value) :
  is_tagged_immediate v = true -> ∃ c : const, v = Value_const c.
Proof.
  unfold is_tagged_immediate. destruct v; cbn; try discriminate.
  intros _. eexists. reflexivity.
Qed.

Lemma unary_number (statics : static_env) (p : unary_primitive)
    (arg v : value) :
  match p with
  | Block_load _ _ _ | Opaque_identity _ | Box_number _ => False
  | _ => True
  end ->
  eval_unary_primitive statics p arg = Some v ->
  ∃ c : const, v = Value_const c.
Proof.
  destruct p as [access mut field | variant_only | | | length_kind | | kind
                | kind op | src dst | | kind | kind | | | reinterpret];
    cbn [eval_unary_primitive]; intros Hp; try contradiction.
  - destruct (eval_is_int statics variant_only arg); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (eval_is_null statics arg); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (block_of statics arg) as [[tag fields] | ]; [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (array_of statics arg) as [elements | ]; [ | discriminate ].
    unfold array_length_of. destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (string_of statics arg) as [bytes | ]; [ | discriminate ].
    destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value kind arg); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value src arg); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value Standard_tagged_immediate arg) as [i | ];
      [ | discriminate ].
    destruct (i =? 0)%Z; [ intros [= <-]; eexists; reflexivity | ].
    destruct (i =? 1)%Z; [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (boxed_number_of statics arg) as [c | ]; [ | discriminate ].
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value Standard_tagged_immediate arg); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value Standard_naked_immediate arg); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct reinterpret; cbn [eval_reinterpret_64_bit_word];
      (destruct (int_of_value _ arg); [ | discriminate ]);
      intros [= <-]; eexists; reflexivity.
Qed.

Lemma immediate_load (statics : static_env) (tag size : option Z)
    (mut : mutability) (field : Z) (arg v : value) :
  eval_unary_primitive statics
    (Block_load (Block_access_values tag size Immediate) mut field) arg
  = Some v ->
  ∃ c : const, v = Value_const c.
Proof.
  cbn [eval_unary_primitive]. unfold eval_block_load.
  destruct (block_of statics arg) as [[block_tag fields] | ];
    [ | discriminate ].
  destruct (_ && _); [ | discriminate ].
  destruct (nth_error fields (Z.to_nat field)) as [loaded | ];
    [ | discriminate ].
  destruct (is_tagged_immediate loaded) eqn:Himmediate; [ | discriminate ].
  intros [= <-]. exact (tagged_const loaded Himmediate).
Qed.

Lemma binary_number (statics : static_env) (p : binary_primitive)
    (arg1 arg2 v : value) :
  match p with
  | Array_load _ _ _ => False
  | _ => True
  end ->
  eval_binary_primitive statics p arg1 arg2 = Some v ->
  ∃ c : const, v = Value_const c.
Proof.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive]; intros Hp; try contradiction.
  - unfold eval_string_load.
    destruct (string_of statics arg1) as [bytes | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_nativeint arg2) as [index | ];
      [ | discriminate ].
    cbv zeta. destruct (_ && _); [ | discriminate ].
    destruct width; intros [= <-]; eexists; reflexivity.
  - destruct (eval_phys_equal statics arg1 arg2); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    destruct (eval_int_arith (int_width kind) op a b); [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate arg2) as [amount | ];
      [ | discriminate ].
    destruct ((0 <=? amount)%Z && (amount <? shift_bound kind)%Z);
      [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
  - destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    intros [= <-]. eexists. reflexivity.
Qed.

Lemma immediates_load (statics : static_env) (kind : array_kind)
    (mut : mutability) (arg1 arg2 v : value) :
  eval_binary_primitive statics (Array_load kind Load_immediates mut) arg1
    arg2
  = Some v ->
  ∃ c : const, v = Value_const c.
Proof.
  cbn [eval_binary_primitive].
  destruct (array_of statics arg1) as [elements | ]; [ | discriminate ].
  unfold array_loaded.
  destruct (array_index elements arg2) as [i | ]; [ | discriminate ].
  destruct (nth_error elements i) as [w | ]; [ | discriminate ].
  cbn [array_kind_of_load_kind array_element_ok].
  destruct (is_tagged_immediate w) eqn:Himmediate; [ | discriminate ].
  intros [= <-]. exact (tagged_const w Himmediate).
Qed.

(* ======================================================================== *)
(* The lifetime of what a [Let] binds is one of its state                   *)
(* ======================================================================== *)

Lemma shorter_region (open : list variable) (a b : lifetime) (r : variable) :
  shorter open a b = Of_region r -> a = Of_region r ∨ b = Of_region r.
Proof.
  destruct a as [ | | ra | ], b as [ | | rb | ]; cbn [shorter];
    try discriminate; try (intros H; left; exact H);
    try (intros H; right; exact H).
  destruct (among ra open && among rb open); [ | discriminate ].
  destruct (among ra (down_to rb open)); intros H;
    [ left; exact H | right; exact H ].
Qed.

Lemma shortest_region (cs : check_state) (ls : list lifetime) (r : variable) :
  shortest cs ls = Of_region r -> In (Of_region r) ls.
Proof.
  unfold shortest. induction ls as [ | l ls IH]; cbn [fold_right];
    [ discriminate | ].
  intros H. apply shorter_region in H as [-> | H];
    [ left; reflexivity | right; exact (IH H) ].
Qed.

Definition env_scoped (cs : check_state) : bool :=
  forallb
    (fun '(_, l) =>
       match l with
       | Of_region r => among r (cs_regions cs)
       | Forever | Of_caller | Gone => true
       end)
    (cs_env cs).

Lemma lifetime_of_scoped (cs : check_state) (s : simple) (r : variable) :
  env_scoped cs = true ->
  lifetime_of cs s = Of_region r ->
  among r (cs_regions cs) = true.
Proof.
  intros Hscoped. destruct s as [x | sym | c]; cbn [lifetime_of];
    try discriminate.
  destruct (lookup x (cs_env cs)) as [l | ] eqn:Hl; [ | discriminate ].
  intros ->.
  exact
    (lookup_forallb
       (fun l =>
          match l with
          | Of_region r => among r (cs_regions cs)
          | Forever | Of_caller | Gone => true
          end)
       x (Of_region r) (cs_env cs) Hscoped Hl).
Qed.

Lemma shortest_scoped (cs : check_state) (simples : list simple)
    (r : variable) :
  env_scoped cs = true ->
  shortest cs (map (lifetime_of cs) simples) = Of_region r ->
  among r (cs_regions cs) = true.
Proof.
  intros Hscoped H. apply shortest_region in H.
  apply in_map_iff in H as (s & Hs & _).
  exact (lifetime_of_scoped cs s r Hscoped Hs).
Qed.

Lemma named_scoped (cs : check_state) (d : named) (r : variable) :
  env_scoped cs = true ->
  named_lifetime cs d = Of_region r ->
  among r (cs_regions cs) = true.
Proof.
  intros Hscoped.
  assert (Hslots :
    ∀ slots : list (value_slot * simple),
      shortest cs (map (fun '(_, s) => lifetime_of cs s) slots) = Of_region r ->
      among r (cs_regions cs) = true).
  { intros slots H. apply shortest_region in H.
    apply in_map_iff in H as ([slot s] & Hs & _).
    exact (lifetime_of_scoped cs s r Hscoped Hs). }
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | from slot closure | from to closure];
    cbn [named_lifetime];
    try exact (lifetime_of_scoped cs _ r Hscoped);
    try exact (Hslots _).
  destruct p as [u | b | v | h | rp].
  - destruct u as [[tag size [ | ]] mut field | | | | | | kind | | | | | | |
                  | ];
      try discriminate;
      exact (shortest_scoped cs args r Hscoped).
  - destruct b as [kind [ | | ] mut | | | | | ]; try discriminate;
      destruct args as [ | a1 [ | a2 [ | a3 args]]];
      first
        [ exact (lifetime_of_scoped cs _ r Hscoped)
        | exact (shortest_scoped cs _ r Hscoped) ].
  - discriminate.
  - exact (shortest_scoped cs args r Hscoped).
  - discriminate.
Qed.

(* ======================================================================== *)
(* Values that an allocation makes                                          *)
(* ======================================================================== *)

Lemma made_all (ok : symbol -> bool) (v : value) :
  is_made v = true ->
  (∀ w : value, In w (value_parts v) -> value_all ok w = true) ->
  value_all ok v = true.
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c | l
       | callee args args_arity];
    cbn [is_made value_parts value_all]; try discriminate; intros _ H;
    try reflexivity.
  - apply forallb_forall. intros [vslot w] Hin. apply H. apply in_map_iff.
    exists (vslot, w). split; [ reflexivity | exact Hin ].
  - apply forallb_forall. intros [vslot w] Hin. apply H. apply in_map_iff.
    exists (vslot, w). split; [ reflexivity | exact Hin ].
  - apply forallb_forall. exact H.
  - apply forallb_forall. exact H.
Qed.

Lemma made_plain (ok : symbol -> bool) (v : value) :
  is_made v = true -> value_plain ok v = value_all ok v.
Proof. destruct v; cbn [is_made]; try discriminate; reflexivity. Qed.

Lemma made_paps (ok : value -> list value -> arity -> bool) (v : value) :
  is_made v = true ->
  (∀ w : value, In w (value_parts v) -> value_paps ok w = true) ->
  value_paps ok v = true.
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c | l
       | callee args args_arity];
    cbn [is_made value_parts value_paps]; try discriminate; intros _ H;
    try reflexivity.
  - apply forallb_forall. intros [vslot w] Hin. apply H. apply in_map_iff.
    exists (vslot, w). split; [ reflexivity | exact Hin ].
  - apply forallb_forall. intros [vslot w] Hin. apply H. apply in_map_iff.
    exists (vslot, w). split; [ reflexivity | exact Hin ].
  - apply forallb_forall. exact H.
  - apply forallb_forall. exact H.
Qed.

Lemma paps_parts (ok : value -> list value -> arity -> bool) (v w : value) :
  value_paps ok v = true -> In w (value_parts v) -> value_paps ok w = true.
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c | l
       | callee args args_arity];
    cbn [value_parts value_paps]; try contradiction.
  - intros H Hin. apply in_map_iff in Hin as ([vslot w0] & <- & Hin).
    exact (forallb_in _ _ (vslot, w0) H Hin).
  - intros H Hin. apply in_map_iff in Hin as ([vslot w0] & <- & Hin).
    exact (forallb_in _ _ (vslot, w0) H Hin).
  - intros H Hin. exact (forallb_in _ _ w H Hin).
  - intros H Hin. exact (forallb_in _ _ w H Hin).
  - intros _ [<- | []]. reflexivity.
Qed.

(* The values that what a symbol is bound to holds directly, as a
   property of each and as a list. *)
Lemma allocated_values_in (P : value -> Prop)
    (allocated : statically_allocated_value) :
  allocated_values P allocated
  <-> ∀ w : value,
        In w (FlambdaRegionsReadsProofs.allocated_values allocated) -> P w.
Proof.
  destruct allocated;
    cbn [allocated_values FlambdaRegionsReadsProofs.allocated_values];
    rewrite ?Forall_forall.
  all: try (split; [ intros _ w [] | intros _; exact I ]).
  all: try (split;
            [ intros H w Hin; apply in_map_iff in Hin as (binding & <- & Hin);
              exact (H binding Hin)
            | intros H binding Hin; apply H;
              exact (in_map snd _ binding Hin) ]).
  all: try reflexivity.
  split;
    [ intros H w [<- | []]; exact H | intros H; apply H; left; reflexivity ].
Qed.

Section Named.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hordered : frames_ordered rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

Hypothesis Hgood_statics :
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    (static_values (live_statics rs)).

Variable vars : env.

Variable cs : check_state.

Variable ms : list mark.

Hypothesis Hrunning : running_at rs vars cs ms.

Hypothesis Hgood_vars : env_values_all (good cert codes rs) vars.

Local Abbreviation holds := (lifetime_holds rs (cs_open cs) ms None).

Lemma running_marked : Forall2 (marked vars) (cs_open cs) ms.
Proof.
  destruct Hrunning as (H & _). exact (proj1 (open_marks_marked _ _ _) H).
Qed.

Lemma running_placed : marks_placed (rs_frames rs) ms = true.
Proof. destruct Hrunning as (_ & H & _). exact H. Qed.

(* The value of a simple is as its lifetime says. *)
Lemma simple_holds (s : simple) (v : value) :
  simple_below (rs_base rs) s = true ->
  eval_simple vars s = Some v ->
  holds (lifetime_of cs s) v.
Proof.
  destruct Hrunning as (_ & _ & _ & Henv).
  destruct s as [x | sym | c]; cbn [lifetime_of eval_simple simple_below].
  - intros _ Hv. destruct (lookup x (cs_env cs)) as [l | ] eqn:Hl;
      [ exact (Henv x l v Hl Hv) | exact I ].
  - intros Hbelow [= <-]. exact (below_base_heap rs sym Hbelow).
  - intros _ [= <-]. reflexivity.
Qed.

Lemma simple_good (s : simple) (v : value) :
  eval_simple vars s = Some v -> good cert codes rs v.
Proof.
  destruct s as [x | sym | c]; cbn [eval_simple].
  - exact (env_values_all_lookup _ vars x v Hgood_vars).
  - intros [= <-]. split; reflexivity.
  - intros [= <-]. split; reflexivity.
Qed.

(* The value of a simple that lives for ever reaches symbols of the
   heap only. *)
Lemma simple_forever (s : simple) (v : value) :
  simple_below (rs_base rs) s = true ->
  is_forever (lifetime_of cs s) = true ->
  eval_simple vars s = Some v ->
  value_all (heap_symbol rs) v = true.
Proof.
  intros Hbelow Hforever Hv. pose proof (simple_holds s v Hbelow Hv) as H.
  destruct (lifetime_of cs s); try discriminate. exact H.
Qed.

(* What a [Let] that does not allocate binds is as its lifetime
   says. *)
Lemma named_holds (d : named) (v : value) :
  is_allocation d = false ->
  (∀ s : simple,
     In s (simples_of_named d) -> simple_below (rs_base rs) s = true) ->
  eval_named (live_statics rs) vars d = Some v ->
  holds (named_lifetime cs d) v.
Proof.
  intros Hnot Hbelow Heval.
  assert (Hnumber : ∀ c : const, holds Forever (Value_const c))
    by (intros c; reflexivity).
  assert (Hread :
    ∀ closure : simple,
      simples_of_named d = [closure] ->
      holds (lifetime_of cs closure) v).
  { intros closure Hsimples.
    apply (holds_named rs Hfresh Hordered Hdeep Hstatics vars (cs_open cs) ms
             running_marked running_placed (lifetime_of cs closure) vars d v
             Hnot);
      [ | exact Heval ].
    intros s w Hin Hw. rewrite Hsimples in Hin. destruct Hin as [<- | []].
    apply (simple_holds closure w); [ | exact Hw ].
    apply Hbelow. rewrite Hsimples. left. reflexivity. }
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | from slot closure | from to closure];
    try discriminate Hnot; try exact (Hread _ eq_refl).
  cbn [eval_named simples_of_named] in *.
  destruct (eval_simples vars args) as [vs | ] eqn:Hargs; [ | discriminate ].
    cbn in Heval. pose proof (eval_simples_Forall2 _ _ _ Hargs) as Hvalues.
    destruct (eval_prim_arity _ _ _ _ Heval)
      as [(p' & v1 & -> & -> & Hp)
         | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]].
    + inversion Hvalues as [ | a1 w1 args' vs' Ha1 Hrest]; subst.
      inversion Hrest; subst.
      assert (Hv1 : holds (shortest cs [lifetime_of cs a1]) v1).
      { apply (holds_rank rs vars _ _ (lifetime_of cs a1) _ v1 running_marked
                 running_placed);
          [ exact (rank_shortest_le cs [lifetime_of cs a1] (lifetime_of cs a1)
                     (or_introl eq_refl)) | ].
        exact (simple_holds a1 v1 (Hbelow a1 (or_introl eq_refl)) Ha1). }
      destruct p' as [[tag size [ | ]] mut field | | | | | | kind | | | | | |
                     | | ];
        cbn [named_lifetime map];
        try discriminate Hnot;
        try (eapply unary_number in Hp as [c ->];
             [ exact (Hnumber c) | exact I ]).
      * refine (holds_unary rs Hfresh Hordered Hdeep Hstatics vars (cs_open cs)
                  ms running_marked running_placed _ _ v1 v _ Hv1 Hp).
        intros kind0 H; discriminate H.
      * destruct (immediate_load _ _ _ _ _ _ _ Hp) as [c ->].
        exact (Hnumber c).
      * refine (holds_unary rs Hfresh Hordered Hdeep Hstatics vars (cs_open cs)
                  ms running_marked running_placed _ _ v1 v _ Hv1 Hp).
        intros kind0 H; discriminate H.
    + inversion Hvalues as [ | a1 w1 args' vs' Ha1 Hrest]; subst.
      inversion Hrest as [ | a2 w2 args'' vs'' Ha2 Hrest']; subst.
      inversion Hrest'; subst.
      destruct p' as [kind [ | | ] mut | | | | | ];
        cbn [named_lifetime];
        try (eapply binary_number in Hp as [c ->];
             [ exact (Hnumber c) | exact I ]).
      * destruct (immediates_load _ _ _ _ _ _ Hp) as [c ->].
        exact (Hnumber c).
      * exact (holds_binary rs Hfresh Hordered Hdeep Hstatics vars (cs_open cs)
                 ms running_marked running_placed _ _ v1 v2 v
                 (simple_holds a1 v1 (Hbelow a1 (or_introl eq_refl)) Ha1) Hp).
      * exact (holds_binary rs Hfresh Hordered Hdeep Hstatics vars (cs_open cs)
                 ms running_marked running_placed _ _ v1 v2 v
                 (simple_holds a1 v1 (Hbelow a1 (or_introl eq_refl)) Ha1) Hp).
    + destruct p'; discriminate Hnot.
Qed.

(* What a [Let] that does not allocate binds is good. *)
Lemma named_good (d : named) (v : value) :
  is_allocation d = false ->
  eval_named (live_statics rs) vars d = Some v ->
  good cert codes rs v.
Proof.
  intros Hnot.
  apply (eval_named_reads (good cert codes rs) (live_statics rs));
    try exact Hnot.
  - intros c _. split; reflexivity.
  - intros w0 w [Hplain Hfits] Hin. split.
    + destruct w0
        as [c | code slots | slot decls slots | s | tag fields | elements | c
           | l | callee args args_arity];
        try contradiction;
        apply all_plain;
        exact (value_all_parts _ _ w Hplain Hin).
    + exact (paps_parts _ w0 w Hfits Hin).
  - intros slot slot' decls slots H. exact H.
  - intros s allocated _ Hlookup. split.
    + apply allocated_values_in.
      exact (proj1 (Forall_forall _ _) Hgood_statics (s, allocated)
               (lookup_in _ _ _ Hlookup)).
    + intros s' _. split; reflexivity.
  - intros s w _. exact (simple_good s w).
Qed.

(* What an allocation of values that live for ever makes is good, and
   reaches symbols of the heap only. *)
Lemma made_good (d : named) (v : value) :
  is_allocation d = true ->
  (∀ s : simple,
     In s (simples_of_named d) -> simple_below (rs_base rs) s = true) ->
  forallb (fun s => is_forever (lifetime_of cs s)) (simples_of_named d)
  = true ->
  eval_named (live_statics rs) vars d = Some v ->
  good cert codes rs v ∧ value_all (heap_symbol rs) v = true.
Proof.
  intros His Hbelow Hforever Heval.
  destruct (eval_named_makes _ _ _ _ His Heval) as [Hmade Hparts].
  assert (Hheap : value_all (heap_symbol rs) v = true).
  { apply (made_all _ v Hmade). intros w Hin.
    destruct (Hparts w Hin) as (s & Hs & Hw).
    exact
      (simple_forever s w (Hbelow s Hs) (forallb_in _ _ s Hforever Hs) Hw). }
  split; [ | exact Hheap ]. split.
  - rewrite (made_plain _ v Hmade). exact Hheap.
  - apply (made_paps _ v Hmade). intros w Hin.
    destruct (Hparts w Hin) as (s & _ & Hw). exact (proj2 (simple_good s w Hw)).
Qed.

End Named.

(* ======================================================================== *)
(* The bound of the newest region                                           *)
(* ======================================================================== *)

(* A value that is live is within the counter of the machine. *)
Lemma within_next (rs : region_state) (v : value) :
  symbols_fresh rs = true ->
  value_within rs None v = true ->
  value_within rs (Some (rs_next rs)) v = true.
Proof.
  intros Hfresh. apply value_all_mono. intros s. unfold symbol_within.
  destruct (heap_symbol rs s); [ reflexivity | ]. cbn [orb before].
  rewrite andb_true_r. intros Hlocal. rewrite Hlocal. cbn [andb].
  apply symbols_fresh_parts in Hfresh as (_ & _ & _ & _ & Hdrawn & _).
  unfold local_in in Hlocal.
  destruct (lookup s (live_locals (rs_frames rs))) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  apply lookup_in in Hs. unfold live_locals in Hs.
  apply in_flat_map in Hs as (f & Hf & Hin).
  destruct (Hdrawn f Hf) as (_ & _ & Hlocals).
  apply Pos.ltb_lt. exact (proj1 (proj2 (Hlocals s allocated Hin))).
Qed.

(* A value as a lifetime says with no bound of the newest region is so
   with the counter of the machine as the bound. *)
Lemma holds_next (rs : region_state) (opens : list variable) (ms : list mark)
    (l : lifetime) (v : value) :
  symbols_fresh rs = true ->
  lifetime_holds rs opens ms None l v ->
  lifetime_holds rs opens ms (Some (rs_next rs)) l v.
Proof.
  intros Hfresh. destruct l as [ | | r | ]; cbn [lifetime_holds];
    try (intros H; exact H).
  - unfold caller_bound. destruct (lowest ms) as [b | ];
      [ intros H; exact H | exact (within_next rs v Hfresh) ].
  - rewrite (region_place_above opens ms (Some (rs_next rs)) r).
    destruct (region_place opens ms None r) as [[m [b | ]] | ];
      try (intros H; exact H).
    intros H Hstack. exact (within_next rs v Hfresh (H Hstack)).
Qed.
