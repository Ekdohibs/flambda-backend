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

(* Proofs of the statements of FlambdaRegionsErasureSpec.v. Nothing here
   needs review.

   The relation between a state of the machine of regions and a state
   that runs the erased term, and what a step that both machines make
   does to it, are in FlambdaRegionsErasedProofs.v and
   FlambdaRegionsPlainStepProofs.v. Here are the steps of regions and
   the two runs.

   - The begin of a region: the erased term has no [Let] for it, so the
     second state makes no step ([begin_region_erased]).
   - The end of a region: one step. The frames that are cut bound
     symbols; what was related stays related for the values that the
     state after may evaluate ([end_region_erased], on the section
     [Cut]). This is where the invariant of
     FlambdaRegionsInvariantSpec.v is used whole, with the fact that
     the frames bind each symbol once
     (FlambdaRegionsOnceProofs.locals_once).
   - A local allocation: one step of the second state, or two
     ([local_allocation_erased], on the section [Allocated]).
   - A region given to a continuation: one step, in which the erased
     handler binds the arguments that are not regions
     ([region_apply_cont_erased]). No value is of the kind [Region]
     (FlambdaRegionsInvariantSpec.no_poison), so what a region
     parameter is given is a mark.
   - Everywhere else the step is that of
     FlambdaRegionsPlainStepProofs.region_step_shared.

   So a step of the machine of regions is answered by no step, one step
   or two of the second state ([region_step_simulated]), and the two
   statements follow by induction on the runs: [erasure_exits] on the
   steps of the first, [erasure_adds_no_exit] on those of the second,
   with the [Let]s at the head of the first expression ([let_depth])
   bounding the steps that the second does not answer. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsInvariantSpec FlambdaRegionsErasureSpec
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsNoPoisonProofs
  FlambdaRegionsOnceProofs FlambdaRegionsInvariantProofs
  FlambdaRegionsErasedProofs FlambdaRegionsPlainStepProofs.
Import ListNotations.

(* The static environment that [step] is given is not unfolded: the
   rules of FlambdaRegionsErasureSpec.value_erased name it. *)
#[local] Opaque live_statics.

(* ======================================================================== *)
(* From one state of the machine of regions to another                      *)
(* ======================================================================== *)

(* FlambdaRegionsErasedProofs.region_extends is of two states that bind
   the same symbols. A local allocation binds new ones, and the end of
   a region unbinds some. What is related in the first state is
   related in the second for the values of a set [G] that the second
   state reads as the first does: a symbol of [G] that the first binds
   is bound alike in the second, to something that holds values of
   [G]; and a part of a partial application of [G] that may be
   evaluated in the second may be in the first, and is of [G]. *)
Section Transfer.

Variable rs rs1 : region_state.
Variable G : value -> Prop.

Hypothesis Hbase : rs_base rs1 = rs_base rs.
Hypothesis G_parts : ∀ v w : value, G v -> In w (value_parts v) -> G w.
Hypothesis G_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity) (part : value),
    G (Value_partial_application callee args args_arity) ->
    part = callee ∨ In part args ->
    value_usable rs1 part = true ->
    value_usable rs part = true ∧ G part.
Hypothesis G_bound :
  ∀ (s : symbol) (allocated : statically_allocated_value),
    G (Value_symbol s) ->
    drawn rs s = true ->
    lookup s (static_values (live_statics rs)) = Some allocated ->
    lookup s (static_values (live_statics rs1)) = Some allocated
    ∧ (∀ w : value, In w (allocated_values allocated) -> G w)
    ∧ (∀ (s' : symbol) (allocated' : statically_allocated_value),
        In s' (allocated_symbols allocated) ->
        lookup s' (static_values (live_statics rs)) = Some allocated' ->
        lookup s' (static_values (live_statics rs1)) = Some allocated').

Lemma drawn_transfer (s : symbol) : drawn rs1 s = drawn rs s.
Proof. unfold drawn. rewrite Hbase. reflexivity. Qed.

Fixpoint value_erased_transfer (v v' : value) (H : value_erased rs v v')
    {struct H} :
  G v -> value_erased rs1 v v'.
Proof.
  intros Hg. erased_cases H.
  - apply Erased_const.
  - apply Erased_boxed_number.
  - apply Erased_location.
  - apply Erased_symbol. rewrite drawn_transfer. exact Hnot_drawn.
  - apply Erased_block.
    assert (Hall : ∀ w : value, In w fields -> G w)
      by (intros w Hin; exact (G_parts _ w Hg Hin)).
    clear Hg. revert fields fields' Hfields Hall. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail] Hall; constructor;
      [ exact (value_erased_transfer _ _ Hhead (Hall w (or_introl eq_refl)))
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - apply Erased_array.
    assert (Hall : ∀ w : value, In w items -> G w)
      by (intros w Hin; exact (G_parts _ w Hg Hin)).
    clear Hg. revert items items' Hitems Hall. fix IH 3.
    intros items items' [ | w w' rest rest' Hhead Htail] Hall; constructor;
      [ exact (value_erased_transfer _ _ Hhead (Hall w (or_introl eq_refl)))
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - apply Erased_closure.
    assert (Hall : ∀ w : value, In w (map snd slots) -> G w)
      by (intros w Hin; exact (G_parts _ w Hg Hin)).
    clear Hg. revert slots slots' Hslots Hall. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail] Hall;
      constructor;
      [ split;
        [ exact Hkey
        | exact
            (value_erased_transfer _ _ Hhead (Hall (snd w) (or_introl eq_refl)))
        ]
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - apply Erased_closure_of_set.
    assert (Hall : ∀ w : value, In w (map snd slots) -> G w)
      by (intros w Hin; exact (G_parts _ w Hg Hin)).
    clear Hg. revert slots slots' Hslots Hall. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail] Hall;
      constructor;
      [ split;
        [ exact Hkey
        | exact
            (value_erased_transfer _ _ Hhead (Hall (snd w) (or_introl eq_refl)))
        ]
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - apply Erased_partial_application.
    + intros Husable1.
      destruct
        (G_partial pcallee pargs parity pcallee Hg (or_introl eq_refl)
           Husable1)
        as [Husable Hgc].
      exact (value_erased_transfer _ _ (Hpcallee Husable) Hgc).
    + assert (Hall :
        ∀ part : value,
          In part pargs ->
          value_usable rs1 part = true ->
          value_usable rs part = true ∧ G part)
        by (intros part Hin;
            exact (G_partial pcallee pargs parity part Hg (or_intror Hin))).
      clear Hpcallee Hg. revert pargs pargs' Hpargs Hall. fix IH 3.
      intros args args' [ | w w' rest rest' [Hkind Hhead] Htail] Hall;
        constructor;
        [ | exact (IH _ _ Htail (fun part Hin => Hall part (or_intror Hin))) ].
      split; [ exact Hkind | ].
      intros Husable1.
      destruct (Hall w (or_introl eq_refl) Husable1) as [Husable Hgw].
      exact (value_erased_transfer _ _ (Hhead Husable) Hgw).
  - destruct (G_bound s _ Hg Hdrawn Hlookup) as (Hlookup1 & Hall & _).
    apply (Erased_local_block rs1 s tag fields fields');
      [ rewrite drawn_transfer; exact Hdrawn | exact Hlookup1 | ].
    simpl in Hall. clear Hlookup Hlookup1 Hg.
    revert fields fields' Hfields Hall. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail] Hall; constructor;
      [ exact (value_erased_transfer _ _ Hhead (Hall w (or_introl eq_refl)))
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - destruct (G_bound s _ Hg Hdrawn Hlookup) as (Hlookup1 & _ & _).
    apply Erased_local_boxed_number;
      [ rewrite drawn_transfer; exact Hdrawn | exact Hlookup1 ].
  - destruct (G_bound s _ Hg Hdrawn Hlookup) as (Hlookup1 & Hall & _).
    apply (Erased_local_array rs1 s items items');
      [ rewrite drawn_transfer; exact Hdrawn | exact Hlookup1 | ].
    simpl in Hall. clear Hlookup Hlookup1 Hg.
    revert items items' Hitems Hall. fix IH 3.
    intros items items' [ | w w' rest rest' Hhead Htail] Hall; constructor;
      [ exact (value_erased_transfer _ _ Hhead (Hall w (or_introl eq_refl)))
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - destruct (G_bound s _ Hg Hdrawn Hlookup) as (Hlookup1 & Hall & _).
    apply (Erased_local_closure rs1 s code slots slots');
      [ rewrite drawn_transfer; exact Hdrawn | exact Hlookup1 | ].
    simpl in Hall. clear Hlookup Hlookup1 Hg.
    revert slots slots' Hslots Hall. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail] Hall;
      constructor;
      [ split;
        [ exact Hkey
        | exact
            (value_erased_transfer _ _ Hhead (Hall (snd w) (or_introl eq_refl)))
        ]
      | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
  - pose proof (Hbound fslot s (lookup_in _ _ _ Hsym)) as Hlookup.
    destruct (G_bound s _ Hg Hdrawn Hlookup) as (_ & Hall & Hsymbols).
    apply (Erased_local_closure_of_set rs1 s fslot syms decls slots slots');
      [ rewrite drawn_transfer; exact Hdrawn | exact Hsym | exact Hsyms | | ].
    + intros slot0 s0 Hin.
      apply (Hsymbols s0); [ | exact (Hbound slot0 s0 Hin) ].
      simpl. apply in_map_iff. exists (slot0, s0).
      split; [ reflexivity | exact Hin ].
    + simpl in Hall. clear Hbound Hlookup Hsymbols Hg.
      revert slots slots' Hslots Hall. fix IH 3.
      intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail] Hall;
        constructor;
        [ split;
          [ exact Hkey
          | exact
              (value_erased_transfer _ _ Hhead
                 (Hall (snd w) (or_introl eq_refl))) ]
        | exact (IH _ _ Htail (fun w0 Hin => Hall w0 (or_intror Hin))) ].
Qed.

Lemma values_erased_transfer (vs vs' : list value) :
  Forall2 (value_erased rs) vs vs' ->
  (∀ w : value, In w vs -> G w) ->
  Forall2 (value_erased rs1) vs vs'.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; intros Hall; constructor;
    [ exact (value_erased_transfer _ _ Hv (Hall v (or_introl eq_refl)))
    | exact (IH (fun w Hin => Hall w (or_intror Hin))) ].
Qed.

Lemma slots_erased_transfer (slots slots' : list (value_slot * value)) :
  Forall2 (slot_erased (value_erased rs)) slots slots' ->
  (∀ w : value, In w (map snd slots) -> G w) ->
  Forall2 (slot_erased (value_erased rs1)) slots slots'.
Proof.
  induction 1 as [ | slot slot' slots slots' [Hkey Hv] _ IH]; intros Hall;
    constructor;
    [ split;
      [ exact Hkey
      | exact
          (value_erased_transfer _ _ Hv (Hall (snd slot) (or_introl eq_refl)))
      ]
    | exact (IH (fun w Hin => Hall w (or_intror Hin))) ].
Qed.

(* A value that the second state may evaluate only if the first may,
   and that is then of [G]. *)
Definition carried (v : value) : Prop :=
  value_usable rs1 v = true -> value_usable rs v = true ∧ G v.

Lemma value_erased_if_usable_transfer (v v' : value) :
  carried v ->
  value_erased_if_usable rs v v' ->
  value_erased_if_usable rs1 v v'.
Proof.
  intros Hc [Hkind Hv]. split; [ exact Hkind | ].
  intros Hu1. destruct (Hc Hu1) as [Hu Hg].
  exact (value_erased_transfer _ _ (Hv Hu) Hg).
Qed.

Lemma values_erased_if_usable_transfer (vs vs' : list value) :
  Forall carried vs ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  Forall2 (value_erased_if_usable rs1) vs vs'.
Proof.
  intros Hall H. revert Hall.
  induction H as [ | v v' vs vs' Hv _ IH]; intros Hall; constructor;
    [ exact (value_erased_if_usable_transfer v v' (Forall_inv Hall) Hv)
    | exact (IH (Forall_inv_tail Hall)) ].
Qed.

Lemma env_erased_transfer (vars vars' : env) :
  env_values_all carried vars ->
  env_erased rs vars vars' ->
  env_erased rs1 vars vars'.
Proof.
  intros Hall Hvars x v Hx Hu1.
  destruct (env_values_all_lookup carried vars x v Hall Hx Hu1) as [Hu Hg].
  destruct (Hvars x v Hx Hu) as (v' & Hx' & Hv).
  exists v'. split; [ exact Hx' | exact (value_erased_transfer _ _ Hv Hg) ].
Qed.

Fixpoint cont_erased_transfer (c c' : cont_value) (H : cont_erased rs c c')
    {struct H} :
  cont_values_all carried c -> cont_erased rs1 c c'.
Proof.
  intros Hall.
  destruct H
    as [ k
       | rconts h vars vars' conts conts' recursive Hh Hrecursive Hagree
           Hvars Hconts
       | c0 c0' extra extra' Htakes Hc0 Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn_takes Hexn_takes
           Hreturn Hexn ].
  - apply Erased_outer.
  - simpl in Hall. destruct Hall as [Hall_vars Hall_conts].
    apply Erased_handler;
      [ exact Hh | exact Hrecursive | exact Hagree
      | exact (env_erased_transfer _ _ Hall_vars Hvars) | ].
    clear Hvars Hagree Hall_vars. revert conts conts' Hconts Hall_conts.
    fix IH 3.
    intros conts conts' [ | kc kc' rest rest' [Hkey Hhead] Htail] Hall_conts;
      constructor;
      [ split;
        [ exact Hkey
        | exact (cont_erased_transfer _ _ Hhead (proj1 Hall_conts)) ]
      | exact (IH _ _ Htail (proj2 Hall_conts)) ].
  - simpl in Hall. destruct Hall as [Hall_extra Hall_c].
    apply Erased_extra_args;
      [ exact Htakes | exact (cont_erased_transfer _ _ Hc0 Hall_c)
      | exact (values_erased_if_usable_transfer _ _ Hall_extra Hextra) ].
  - simpl in Hall. destruct Hall as (Hall_remaining & Hall_return & Hall_exn).
    apply Erased_over_application;
      [ exact (values_erased_if_usable_transfer _ _ Hall_remaining Hremaining)
      | exact Hreturn_takes | exact Hexn_takes
      | exact (cont_erased_transfer _ _ Hreturn Hall_return)
      | exact (cont_erased_transfer _ _ Hexn Hall_exn) ].
Qed.

Lemma conts_erased_transfer (conts conts' : cont_env) :
  conts_values_all carried conts ->
  conts_erased rs conts conts' ->
  conts_erased rs1 conts conts'.
Proof.
  intros Hall H. revert Hall.
  induction H as [ | kc kc' conts conts' [Hkey Hc] _ IH]; intros Hall;
    constructor;
    [ split;
      [ exact Hkey | exact (cont_erased_transfer _ _ Hc (proj1 Hall)) ]
    | exact (IH (proj2 Hall)) ].
Qed.

Lemma allocated_erased_transfer
    (allocated allocated' : statically_allocated_value) :
  allocated_erased rs allocated allocated' ->
  (∀ w : value, In w (allocated_values allocated) -> G w) ->
  allocated_erased rs1 allocated allocated'.
Proof.
  intros H Hall. allocated_erased_cases H; simpl in Hall.
  - apply Erased_static_closure.
    exact (slots_erased_transfer _ _ Haslots Hall).
  - apply Erased_static_closure_of_set;
      [ intros slot s Hin; rewrite drawn_transfer; exact (Hasyms slot s Hin)
      | exact (slots_erased_transfer _ _ Haslots Hall) ].
  - apply Erased_static_block.
    exact (values_erased_transfer _ _ Hafields Hall).
  - apply Erased_static_boxed_number.
    exact (value_erased_transfer _ _ Hav (Hall av (or_introl eq_refl))).
  - apply Erased_static_string.
  - apply Erased_static_imported.
  - apply Erased_static_immutable_array.
    exact (values_erased_transfer _ _ Haelements Hall).
Qed.

(* The second static environment of the relation, for a state that
   has the code of the first, binds the symbols that the machine did
   not draw as the first does, to things that hold values of [G], and
   whose static environment binds what that of the first binds. *)
Lemma statics_erased_transfer (statics' : static_env) :
  static_code (state_statics (rs_state rs1))
  = static_code (state_statics (rs_state rs)) ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
    drawn rs s = false ->
    lookup s (static_values (live_statics rs1)) = Some allocated ->
    lookup s (static_values (live_statics rs)) = Some allocated
    ∧ ∀ w : value, In w (allocated_values allocated) -> G w) ->
  (∀ s : symbol,
    lookup s (static_values (state_statics (rs_state rs))) <> None ->
    lookup s (static_values (state_statics (rs_state rs1))) <> None) ->
  statics_erased rs statics' ->
  statics_erased rs1 statics'.
Proof.
  intros Hcode Hsame Hgrow (Hc & Hreturns & Hsymbols & Hbound).
  unfold statics_erased. rewrite Hcode.
  split; [ exact Hc | ]. split; [ exact Hreturns | ]. split.
  - intros s allocated Hnd1 Hl1. rewrite drawn_transfer in Hnd1.
    destruct (Hsame s allocated Hnd1 Hl1) as [Hl Hall].
    destruct (Hsymbols s allocated Hnd1 Hl) as (allocated' & Hl' & Ha).
    exists allocated'.
    split; [ exact Hl' | exact (allocated_erased_transfer _ _ Ha Hall) ].
  - intros s allocated' Hl'. exact (Hgrow s (Hbound s allocated' Hl')).
Qed.

End Transfer.

(* ======================================================================== *)
(* What the tests of the invariant give                                     *)
(* ======================================================================== *)

Lemma symbols_fresh_tests (rs : region_state) :
  symbols_fresh rs = true ->
  (rs_base rs <= rs_next rs)%positive
  ∧ expr_below (rs_base rs) (state_expr (rs_state rs)) = true
  ∧ forallb (fun '(_, v) => value_below (rs_next rs) v)
      (state_vars (rs_state rs))
    = true
  ∧ forallb (fun '(_, c) => cont_below (rs_base rs) (rs_next rs) c)
      (state_conts (rs_state rs))
    = true
  ∧ forallb
      (fun '(sym, allocated) =>
         symbol_below (rs_next rs) sym
         && allocated_below (rs_next rs) allocated)
      (static_values (state_statics (rs_state rs)))
    = true
  ∧ forallb (frame_fresh (rs_base rs) (rs_next rs)) (rs_frames rs) = true
  ∧ forallb
      (fun '(sym, _) =>
         negb
           (is_some
              (lookup sym (static_values (state_statics (rs_state rs))))))
      (live_locals (rs_frames rs))
    = true.
Proof.
  unfold symbols_fresh. cbv zeta. intros H.
  apply andb_true_iff in H as [H F9]. apply andb_true_iff in H as [H _].
  apply andb_true_iff in H as [H F7]. apply andb_true_iff in H as [H F6].
  apply andb_true_iff in H as [H F5]. apply andb_true_iff in H as [H F4].
  apply andb_true_iff in H as [H _]. apply andb_true_iff in H as [F1 F2].
  apply Pos.leb_le in F1. repeat split; assumption.
Qed.

(* What a frame binds: a symbol that the machine drew, below the
   counter, that the static environment does not bind. *)
Lemma frame_symbol (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value) :
  symbols_fresh rs = true ->
  In (s, allocated) (live_locals (rs_frames rs)) ->
  drawn rs s = true
  ∧ symbol_below (rs_next rs) s = true
  ∧ allocated_below (rs_next rs) allocated = true
  ∧ lookup s (static_values (state_statics (rs_state rs))) = None.
Proof.
  intros Hfresh Hin.
  destruct (symbols_fresh_tests rs Hfresh) as (_ & _ & _ & _ & _ & F7 & F9).
  pose proof (forallb_in _ _ (s, allocated) F9 Hin) as Hunbound.
  cbv beta iota in Hunbound.
  unfold live_locals in Hin. apply in_flat_map in Hin as (f & Hf & Hin).
  pose proof (forallb_in _ _ f F7 Hf) as Hframe. unfold frame_fresh in Hframe.
  apply andb_true_iff in Hframe as [_ Hframe].
  pose proof (forallb_in _ _ (s, allocated) Hframe Hin) as Hsa.
  cbv beta iota in Hsa. apply andb_true_iff in Hsa as [Hbetween Hbelow].
  unfold symbol_between in Hbetween.
  apply andb_true_iff in Hbetween as [Hdrawn Hlt].
  split; [ exact Hdrawn | ]. split; [ exact Hlt | ]. split; [ exact Hbelow | ].
  destruct (lookup s (static_values (state_statics (rs_state rs))));
    [ discriminate | reflexivity ].
Qed.

(* Everything bound is below the counter. *)
Lemma fresh_bound (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value) :
  symbols_fresh rs = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  symbol_below (rs_next rs) s = true
  ∧ allocated_below (rs_next rs) allocated = true.
Proof.
  intros Hfresh Hlookup.
  rewrite live_statics_values, lookup_app in Hlookup.
  destruct (lookup s (live_locals (rs_frames rs))) as [allocated0 | ]
    eqn:Hlocals.
  - injection Hlookup as ->.
    destruct (frame_symbol rs s allocated Hfresh (lookup_in _ _ _ Hlocals))
      as (_ & Hs & Ha & _).
    exact (conj Hs Ha).
  - destruct (symbols_fresh_tests rs Hfresh) as (_ & _ & _ & _ & F6 & _ & _).
    pose proof (forallb_in _ _ (s, allocated) F6 (lookup_in _ _ _ Hlookup))
      as Hsa.
    cbv beta iota in Hsa. apply andb_true_iff in Hsa as [Hs Ha].
    exact (conj Hs Ha).
Qed.

(* A symbol of the heap is in no frame. *)
Lemma heap_lookup (rs : region_state) (s : symbol) :
  symbols_fresh rs = true ->
  heap_symbol rs s = true ->
  lookup s (static_values (live_statics rs))
  = lookup s (static_values (state_statics (rs_state rs))).
Proof.
  intros Hfresh Hheap. rewrite live_statics_values, lookup_app.
  destruct (lookup s (live_locals (rs_frames rs))) as [allocated | ]
    eqn:Hlocals; [ | reflexivity ].
  destruct (frame_symbol rs s allocated Hfresh (lookup_in _ _ _ Hlocals))
    as (Hdrawn & _ & _ & Hunbound).
  unfold heap_symbol in Hheap. rewrite Hdrawn, Hunbound in Hheap.
  discriminate.
Qed.

Lemma value_holds_placed (near ok : symbol -> bool) (w : value) :
  (∀ s : symbol, ok s = true -> near s = true) ->
  value_holds ok w = true ->
  value_placed near ok w = true.
Proof. intros Hok. destruct w; simpl; auto. Qed.

(* ======================================================================== *)
(* Bindings that are bound once                                              *)
(* ======================================================================== *)

Lemma nodup_app_disjoint {A : Type} (l1 l2 : list A) (x : A) :
  NoDup (l1 ++ l2) -> In x l1 -> In x l2 -> False.
Proof.
  induction l1 as [ | a l1 IH]; simpl; [ intros _ [] | ].
  intros Hn [<- | Hin1] Hin2; inversion Hn as [ | a' l' Hnotin Hrest]; subst.
  - apply Hnotin. apply in_or_app. right. exact Hin2.
  - exact (IH Hrest Hin1 Hin2).
Qed.

(* The bindings behind others that share no name with them, nor with
   what follows, are found as they are without those others. *)
Lemma lookup_suffix {A : Type} (name : positive)
    (front back rest : list (positive * A)) (a : A) :
  NoDup (map fst (front ++ back)) ->
  (∀ (k : positive) (v : A),
    In (k, v) (front ++ back) -> lookup k rest = None) ->
  lookup name (back ++ rest) = Some a ->
  lookup name ((front ++ back) ++ rest) = Some a.
Proof.
  intros Hnodup Hdisjoint. rewrite !lookup_app.
  destruct (lookup name back) as [b | ] eqn:Hback.
  - intros [= <-].
    rewrite (lookup_not_in name front); [ reflexivity | ].
    intros v Hin. rewrite map_app in Hnodup.
    apply (nodup_app_disjoint _ _ name Hnodup).
    + apply in_map_iff. exists (name, v). split; [ reflexivity | exact Hin ].
    + apply in_map_iff. exists (name, b).
      split; [ reflexivity | exact (lookup_in _ _ _ Hback) ].
  - intros Hrest.
    destruct (lookup name front) as [v | ] eqn:Hfront; [ | exact Hrest ].
    rewrite
      (Hdisjoint name v
         (in_or_app _ _ _ (or_introl (lookup_in _ _ _ Hfront))))
      in Hrest.
    discriminate.
Qed.

(* ======================================================================== *)
(* After the end of a region                                                *)
(* ======================================================================== *)

(* The second state has the frames of the first below one of them, and
   its base and static environment. The values carried over are those
   that a variable may hold and that the second state may evaluate: a
   symbol that it may evaluate, or a value that is not a symbol and
   holds symbols of the heap only. *)
Section Cut.

Variable rs rs1 : region_state.
Variable above : list frame.
Variable f : frame.

Hypothesis Hfresh : symbols_fresh rs = true.
Hypothesis Hlive : values_live rs = true.
Hypothesis Honce : locals_once rs.
Hypothesis Hbase : rs_base rs1 = rs_base rs.
Hypothesis Hstatics :
  state_statics (rs_state rs1) = state_statics (rs_state rs).
Hypothesis Hframes : rs_frames rs = above ++ f :: rs_frames rs1.

Lemma cut_locals :
  live_locals (rs_frames rs)
  = (live_locals above ++ frame_locals f) ++ live_locals (rs_frames rs1).
Proof.
  rewrite Hframes. unfold live_locals. rewrite flat_map_app. simpl.
  rewrite app_assoc. reflexivity.
Qed.

Lemma lookup_cut (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values (live_statics rs1)) = Some allocated ->
  lookup s (static_values (live_statics rs)) = Some allocated.
Proof.
  rewrite !live_statics_values, Hstatics, cut_locals.
  apply lookup_suffix.
  - pose proof Honce as Hnodup. unfold locals_once in Hnodup.
    rewrite cut_locals in Hnodup. exact Hnodup.
  - intros k v Hin.
    assert (Hin' : In (k, v) (live_locals (rs_frames rs)))
      by (rewrite cut_locals; exact Hin).
    destruct (frame_symbol rs k v Hfresh Hin') as (_ & _ & _ & Hunbound).
    exact Hunbound.
Qed.

Lemma heap_cut (s : symbol) : heap_symbol rs1 s = heap_symbol rs s.
Proof. unfold heap_symbol, drawn. rewrite Hbase, Hstatics. reflexivity. Qed.

Lemma usable_cut (v : value) :
  value_usable rs1 v = true -> value_usable rs v = true.
Proof.
  destruct v as [ | | | s | | | | | ]; try (intros _; reflexivity).
  unfold value_usable, drawn. rewrite Hbase.
  destruct (negb (Pos.leb (rs_base rs) s)); simpl; [ auto | ].
  destruct (lookup s (static_values (live_statics rs1))) as [allocated | ]
    eqn:Hl; [ | discriminate ].
  intros _. rewrite (lookup_cut s allocated Hl). reflexivity.
Qed.

(* A symbol of the heap is bound in the two states alike. *)
Lemma heap_lookup_cut (s : symbol) :
  heap_symbol rs s = true ->
  lookup s (static_values (live_statics rs1))
  = lookup s (static_values (live_statics rs)).
Proof.
  intros Hheap. rewrite (heap_lookup rs s Hfresh Hheap).
  rewrite live_statics_values, Hstatics, lookup_app.
  destruct (lookup s (live_locals (rs_frames rs1))) as [allocated | ] eqn:Hl;
    [ | reflexivity ].
  exfalso.
  assert (Hin : In (s, allocated) (live_locals (rs_frames rs)))
    by (rewrite cut_locals; apply in_or_app; right;
        exact (lookup_in _ _ _ Hl)).
  destruct (frame_symbol rs s allocated Hfresh Hin)
    as (Hdrawn & _ & _ & Hunbound).
  unfold heap_symbol in Hheap. rewrite Hdrawn, Hunbound in Hheap.
  discriminate.
Qed.

(* So is a symbol of a frame that is left. *)
Lemma near_lookup_cut (s : symbol) (allocated : statically_allocated_value) :
  heap_symbol rs s || local_in (rs_frames rs1) s = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  lookup s (static_values (live_statics rs1)) = Some allocated.
Proof.
  intros Hnear Hl.
  destruct (heap_symbol rs s) eqn:Hheap.
  - rewrite (heap_lookup_cut s Hheap). exact Hl.
  - simpl in Hnear. unfold local_in in Hnear.
    destruct (lookup s (live_locals (rs_frames rs1))) as [allocated1 | ]
      eqn:Hl1; [ | discriminate ].
    assert (H1 :
      lookup s (static_values (live_statics rs1)) = Some allocated1)
      by (rewrite live_statics_values, lookup_app, Hl1; reflexivity).
    rewrite (lookup_cut s allocated1 H1) in Hl. injection Hl as <-.
    exact H1.
Qed.

Definition cut_good (v : value) : Prop :=
  value_placed (live_symbol rs1) (heap_symbol rs) v = true.

Lemma heap_live_cut (s : symbol) :
  heap_symbol rs s = true -> live_symbol rs1 s = true.
Proof. intros H. apply heap_live. rewrite heap_cut. exact H. Qed.

Lemma holds_cut_good (w : value) :
  value_holds (heap_symbol rs) w = true -> cut_good w.
Proof. exact (value_holds_placed _ _ w heap_live_cut). Qed.

Lemma cut_good_parts (v w : value) :
  cut_good v -> In w (value_parts v) -> cut_good w.
Proof.
  intros Hv Hin.
  destruct v; simpl in Hin; try contradiction;
    apply holds_cut_good;
    exact (forallb_in _ _ w (value_holds_parts _ _ Hv) Hin).
Qed.

Lemma kept_carried (part : value) :
  value_kept (heap_symbol rs) part = true -> carried rs rs1 cut_good part.
Proof.
  intros Hk Hu1. split; [ exact (usable_cut part Hu1) | ].
  destruct part; try exact Hk. exact Hu1.
Qed.

Lemma cut_good_partial (callee : value) (args : list value)
    (args_arity : arity) (part : value) :
  cut_good (Value_partial_application callee args args_arity) ->
  part = callee ∨ In part args ->
  value_usable rs1 part = true ->
  value_usable rs part = true ∧ cut_good part.
Proof.
  intros Hg Hpart. unfold cut_good in Hg. simpl in Hg.
  apply andb_true_iff in Hg as [Hc Ha].
  change (carried rs rs1 cut_good part). apply kept_carried.
  destruct Hpart as [-> | Hin];
    [ exact Hc | exact (forallb_in _ _ part Ha Hin) ].
Qed.

Lemma frames_hold_cut : frames_hold rs (rs_frames rs1) = true.
Proof.
  pose proof (proj1 (values_live_parts rs) Hlive) as (_ & L2 & _ & _).
  rewrite Hframes in L2. apply frames_hold_app in L2. simpl in L2.
  apply andb_true_iff in L2 as [_ L2]. exact L2.
Qed.

Lemma cut_good_bound (s : symbol) (allocated : statically_allocated_value) :
  cut_good (Value_symbol s) ->
  drawn rs s = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  lookup s (static_values (live_statics rs1)) = Some allocated
  ∧ (∀ w : value, In w (allocated_values allocated) -> cut_good w)
  ∧ (∀ (s' : symbol) (allocated' : statically_allocated_value),
      In s' (allocated_symbols allocated) ->
      lookup s' (static_values (live_statics rs)) = Some allocated' ->
      lookup s' (static_values (live_statics rs1)) = Some allocated').
Proof.
  intros Hg Hdrawn Hl.
  pose proof (proj1 (values_live_parts rs) Hlive) as (L1 & _ & _ & _).
  assert (Hl1 : lookup s (static_values (live_statics rs1)) = Some allocated).
  { unfold cut_good in Hg. simpl in Hg.
    unfold live_symbol, value_usable in Hg.
    rewrite (drawn_transfer rs rs1 Hbase s), Hdrawn in Hg. simpl in Hg.
    destruct (lookup s (static_values (live_statics rs1))) as [allocated1 | ]
      eqn:H1; [ | discriminate ].
    rewrite (lookup_cut s allocated1 H1) in Hl. injection Hl as <-.
    reflexivity. }
  split; [ exact Hl1 | ].
  rewrite live_statics_values, lookup_app in Hl1.
  destruct (lookup s (live_locals (rs_frames rs1))) as [allocated1 | ]
    eqn:Hlocals.
  - injection Hl1 as ->.
    pose proof
      (frames_hold_in rs _ s allocated frames_hold_cut
         (lookup_in _ _ _ Hlocals))
      as Hp.
    rewrite allocated_placed_parts in Hp.
    apply andb_true_iff in Hp as [Hvals Hsyms].
    split.
    + intros w Hin.
      refine
        (value_placed_mono _ _ _ _ w _ (fun s0 H0 => H0)
           (forallb_in _ _ w Hvals Hin)).
      intros s' Hs'. apply near_live. rewrite heap_cut. exact Hs'.
    + intros s' allocated' Hin.
      exact (near_lookup_cut s' allocated' (forallb_in _ _ s' Hsyms Hin)).
  - rewrite Hstatics in Hl1.
    pose proof
      (lookup_forallb (allocated_holds (heap_symbol rs)) s allocated _ L1 Hl1)
      as Hh.
    rewrite allocated_holds_parts in Hh.
    apply andb_true_iff in Hh as [Hvals Hsyms].
    split.
    + intros w Hin. exact (holds_cut_good w (forallb_in _ _ w Hvals Hin)).
    + intros s' allocated' Hin. apply near_lookup_cut.
      rewrite (forallb_in _ _ s' Hsyms Hin). reflexivity.
Qed.

Lemma env_erased_cut (vars vars' : env) :
  forallb (fun '(_, v) => value_kept (heap_symbol rs) v) vars = true ->
  env_erased rs vars vars' ->
  env_erased rs1 vars vars'.
Proof.
  intros Hkept.
  apply
    (env_erased_transfer rs rs1 cut_good Hbase cut_good_parts
       cut_good_partial cut_good_bound).
  apply Forall_forall. intros [x v] Hin.
  exact (kept_carried v (forallb_in _ _ (x, v) Hkept Hin)).
Qed.

Lemma conts_erased_cut (conts conts' : cont_env) :
  forallb (fun '(_, c) => cont_kept (heap_symbol rs) c) conts = true ->
  conts_erased rs conts conts' ->
  conts_erased rs1 conts conts'.
Proof.
  intros Hkept.
  apply
    (conts_erased_transfer rs rs1 cut_good Hbase cut_good_parts
       cut_good_partial cut_good_bound).
  apply (conts_values_all_impl (fun v => value_kept (heap_symbol rs) v = true));
    [ exact kept_carried | ].
  exact (proj1 (conts_kept_split (heap_symbol rs) conts) Hkept).
Qed.

Lemma statics_erased_cut (statics' : static_env) :
  statics_erased rs statics' -> statics_erased rs1 statics'.
Proof.
  apply
    (statics_erased_transfer rs rs1 cut_good Hbase cut_good_parts
       cut_good_partial cut_good_bound).
  - rewrite Hstatics. reflexivity.
  - intros s allocated Hnd Hl1.
    pose proof (lookup_cut s allocated Hl1) as Hl. split; [ exact Hl | ].
    pose proof (proj1 (values_live_parts rs) Hlive) as (L1 & _ & _ & _).
    assert (Hheap : heap_symbol rs s = true)
      by (unfold heap_symbol; rewrite Hnd; reflexivity).
    rewrite (heap_lookup rs s Hfresh Hheap) in Hl.
    pose proof
      (lookup_forallb (allocated_holds (heap_symbol rs)) s allocated _ L1 Hl)
      as Hh.
    rewrite allocated_holds_parts in Hh. apply andb_true_iff in Hh as [Hvals _].
    intros w Hin. exact (holds_cut_good w (forallb_in _ _ w Hvals Hin)).
  - rewrite Hstatics. auto.
Qed.

End Cut.

(* ======================================================================== *)
(* After a local allocation                                                 *)
(* ======================================================================== *)

(* The second state binds what the first binds and the new symbols,
   which are from the counter of the first on. The values carried over
   are those below that counter. *)
Section Allocated.

Variable rs rs1 : region_state.
Variable bindings : list (symbol * statically_allocated_value).

Hypothesis Hfresh : symbols_fresh rs = true.
Hypothesis Hbase : rs_base rs1 = rs_base rs.
Hypothesis Hvalues :
  static_values (live_statics rs1)
  = bindings ++ static_values (live_statics rs).
Hypothesis Hcode :
  static_code (state_statics (rs_state rs1))
  = static_code (state_statics (rs_state rs)).
Hypothesis Hgrow :
  ∀ s : symbol,
    lookup s (static_values (state_statics (rs_state rs))) <> None ->
    lookup s (static_values (state_statics (rs_state rs1))) <> None.
Hypothesis Hfrom :
  ∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) bindings -> (rs_next rs <= s)%positive.

Definition below_next (v : value) : Prop :=
  value_below (rs_next rs) v = true.

Lemma lookup_allocated (s : symbol) :
  symbol_below (rs_next rs) s = true ->
  lookup s (static_values (live_statics rs1))
  = lookup s (static_values (live_statics rs)).
Proof.
  intros Hs. rewrite Hvalues, lookup_app.
  rewrite (lookup_not_in s bindings); [ reflexivity | ].
  intros allocated Hin. pose proof (Hfrom s allocated Hin).
  unfold symbol_below in Hs. apply Pos.ltb_lt in Hs. lia.
Qed.

Lemma usable_allocated (v : value) :
  below_next v -> value_usable rs1 v = value_usable rs v.
Proof.
  destruct v as [ | | | s | | | | | ]; try reflexivity.
  unfold below_next. simpl. intros Hs.
  unfold value_usable, drawn. rewrite Hbase, (lookup_allocated s Hs).
  reflexivity.
Qed.

Lemma below_carried (v : value) :
  below_next v -> carried rs rs1 below_next v.
Proof.
  intros Hv Hu1. rewrite (usable_allocated v Hv) in Hu1. exact (conj Hu1 Hv).
Qed.

Lemma below_next_parts (v w : value) :
  below_next v -> In w (value_parts v) -> below_next w.
Proof.
  intros Hv Hin. exact (forallb_in _ _ w (value_below_parts _ v Hv) Hin).
Qed.

Lemma below_next_partial (callee : value) (args : list value)
    (args_arity : arity) (part : value) :
  below_next (Value_partial_application callee args args_arity) ->
  part = callee ∨ In part args ->
  value_usable rs1 part = true ->
  value_usable rs part = true ∧ below_next part.
Proof.
  intros Hg Hpart. unfold below_next in Hg. simpl in Hg.
  apply andb_true_iff in Hg as [Hc Ha].
  change (carried rs rs1 below_next part). apply below_carried.
  destruct Hpart as [-> | Hin];
    [ exact Hc | exact (forallb_in _ _ part Ha Hin) ].
Qed.

Lemma below_next_bound (s : symbol)
    (allocated : statically_allocated_value) :
  below_next (Value_symbol s) ->
  drawn rs s = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  lookup s (static_values (live_statics rs1)) = Some allocated
  ∧ (∀ w : value, In w (allocated_values allocated) -> below_next w)
  ∧ (∀ (s' : symbol) (allocated' : statically_allocated_value),
      In s' (allocated_symbols allocated) ->
      lookup s' (static_values (live_statics rs)) = Some allocated' ->
      lookup s' (static_values (live_statics rs1)) = Some allocated').
Proof.
  intros Hg _ Hl.
  destruct (fresh_bound rs s allocated Hfresh Hl) as [_ Ha].
  rewrite allocated_below_parts in Ha.
  apply andb_true_iff in Ha as [Hvals Hsyms].
  split; [ rewrite (lookup_allocated s Hg); exact Hl | ].
  split; [ intros w Hin; exact (forallb_in _ _ w Hvals Hin) | ].
  intros s' allocated' Hin Hl'.
  rewrite (lookup_allocated s' (forallb_in _ _ s' Hsyms Hin)). exact Hl'.
Qed.

Lemma value_erased_allocated (v v' : value) :
  value_erased rs v v' -> below_next v -> value_erased rs1 v v'.
Proof.
  exact
    (value_erased_transfer rs rs1 below_next Hbase below_next_parts
       below_next_partial below_next_bound v v').
Qed.

Lemma env_erased_allocated (vars vars' : env) :
  forallb (fun '(_, v) => value_below (rs_next rs) v) vars = true ->
  env_erased rs vars vars' ->
  env_erased rs1 vars vars'.
Proof.
  intros Hbelow.
  apply
    (env_erased_transfer rs rs1 below_next Hbase below_next_parts
       below_next_partial below_next_bound).
  apply Forall_forall. intros [x v] Hin.
  exact (below_carried v (forallb_in _ _ (x, v) Hbelow Hin)).
Qed.

Lemma conts_erased_allocated (conts conts' : cont_env) :
  forallb (fun '(_, c) => cont_below (rs_base rs) (rs_next rs) c) conts
  = true ->
  conts_erased rs conts conts' ->
  conts_erased rs1 conts conts'.
Proof.
  intros Hbelow.
  apply
    (conts_erased_transfer rs rs1 below_next Hbase below_next_parts
       below_next_partial below_next_bound).
  apply (conts_values_all_impl (fun v => value_below (rs_next rs) v = true));
    [ exact below_carried | ].
  exact
    (proj2 (proj1 (conts_below_split (rs_base rs) (rs_next rs) conts) Hbelow)).
Qed.

Lemma statics_erased_allocated (statics' : static_env) :
  statics_erased rs statics' -> statics_erased rs1 statics'.
Proof.
  apply
    (statics_erased_transfer rs rs1 below_next Hbase below_next_parts
       below_next_partial below_next_bound).
  - exact Hcode.
  - intros s allocated Hnd Hl1.
    destruct (symbols_fresh_tests rs Hfresh) as (F1 & _).
    assert (Hs : symbol_below (rs_next rs) s = true).
    { unfold drawn in Hnd. apply Pos.leb_gt in Hnd.
      unfold symbol_below. apply Pos.ltb_lt. lia. }
    rewrite (lookup_allocated s Hs) in Hl1. split; [ exact Hl1 | ].
    destruct (fresh_bound rs s allocated Hfresh Hl1) as [_ Ha].
    rewrite allocated_below_parts in Ha. apply andb_true_iff in Ha as [Hvals _].
    intros w Hin. exact (forallb_in _ _ w Hvals Hin).
  - exact Hgrow.
Qed.

End Allocated.

(* ======================================================================== *)
(* What answers a step of the machine of regions                            *)
(* ======================================================================== *)

(* The [Let]s at the head of an expression. A step of regions that the
   erased term has no step for, the begin of a region, goes under one. *)
Fixpoint let_depth (e : expr) : nat :=
  match e with
  | Let _ _ _ body => S (let_depth body)
  | _ => 0
  end.

(* What the second state answers to a result of the machine of
   regions. Nothing is asked if the first has no step. An exit is
   answered by an exit through the same continuation, with the values
   related where they may be evaluated. A step is answered by no step,
   the second state being related to the state after and the first
   expression having one [Let] fewer at its head; or by one step, or
   by two, to a state related. *)
Definition sim_result (rs : region_state) (s' : state)
    (result : region_step_result) : Prop :=
  match result with
  | Region_undefined _ => True
  | Region_exit k vs rs1 =>
    ∃ (vs' : list value) (hp : heap),
      step s' = Exit k vs' hp ∧ Forall2 (value_erased_if_usable rs1) vs vs'
  | Region_step rs1 =>
    (state_erased rs1 s'
     ∧ let_depth (state_expr (rs_state rs1))
       < let_depth (state_expr (rs_state rs)))
    ∨ (∃ s1' : state, step s' = Step s1' ∧ state_erased rs1 s1')
    ∨ (∃ s1' s2' : state,
        step s' = Step s1' ∧ step s1' = Step s2' ∧ state_erased rs1 s2')
  end.

Lemma result_erased_sim (rs : region_state) (s' : state)
    (result : region_step_result) :
  result_erased result (step s') -> sim_result rs s' result.
Proof.
  destruct result as [rs1 | k vs rs1 | fault]; simpl.
  - intros H. right. left. exact H.
  - intros H. exact H.
  - intros _. exact I.
Qed.

(* ======================================================================== *)
(* The begin of a region                                                    *)
(* ======================================================================== *)

(* The counter is bound to nothing. *)
Lemma next_unbound (rs : region_state) :
  symbols_fresh rs = true ->
  lookup (rs_next rs) (static_values (live_statics rs)) = None.
Proof.
  intros Hfresh.
  destruct (lookup (rs_next rs) (static_values (live_statics rs)))
    as [allocated | ] eqn:Hl; [ | reflexivity ].
  destruct (fresh_bound rs _ allocated Hfresh Hl) as [Hs _].
  unfold symbol_below in Hs. apply Pos.ltb_lt in Hs. lia.
Qed.

(* The erased term has no [Let] for it: the second state is related to
   the state after, in which the variable is bound to a mark, which
   may not be evaluated. *)
Lemma begin_region_erased (rs : region_state) (s' : state) (x : variable)
    (kind : flambda_kind) (d : named) (mark_kind : mark_kind) (body : expr) :
  symbols_fresh rs = true ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Let x kind d body ->
  (∀ rconts : region_conts,
    erase rconts (Let x kind d body) = erase rconts body) ->
  sim_result rs s' (begin_region rs x kind mark_kind body).
Proof.
  intros Hfresh
    (rconts & He & Hcalls & Hagree & Hvars & Hconts & Hstatics & Hheap)
    Hexpr Herase.
  unfold begin_region. destruct (kind_eqb kind Region); [ | exact I ].
  left. split; [ | rewrite Hexpr; simpl; lia ].
  pose
    (rs1 :=
       bind_let rs x (Value_symbol (rs_next rs)) body
         (state_statics (rs_state rs))
         (Frame (rs_next rs) mark_kind [] :: rs_frames rs)
         (Pos.succ (rs_next rs)) (rs_counts rs)).
  assert (Hsame : same_statics rs rs1).
  { split; [ reflexivity | ]. split; [ reflexivity | ].
    rewrite !live_statics_values. reflexivity. }
  pose proof (same_statics_extends rs rs1 Hsame) as Hextends.
  exists rconts.
  split; [ rewrite He, Hexpr; exact (Herase rconts) | ].
  split; [ rewrite Hexpr in Hcalls; exact Hcalls | ].
  split; [ exact Hagree | ].
  split.
  { apply
      (env_erased_cons_dead rs1 x (Value_symbol (rs_next rs))
         (state_vars (rs_state rs)) (state_vars s'));
      [ | exact (env_erased_extend rs rs1 _ _ Hextends Hvars) ].
    unfold value_usable, drawn.
    rewrite (proj2 (proj2 Hsame)), (next_unbound rs Hfresh).
    destruct (symbols_fresh_tests rs Hfresh) as (F1 & _).
    apply Pos.leb_le in F1. change (rs_base rs1) with (rs_base rs).
    rewrite F1. reflexivity. }
  split; [ exact (conts_erased_extend rs rs1 _ _ Hextends Hconts) | ].
  split; [ exact (statics_erased_same rs rs1 _ Hsame Hstatics) | exact Hheap ].
Qed.

(* ======================================================================== *)
(* The end of a region                                                      *)
(* ======================================================================== *)

Lemma end_region_no_exit (rs : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) (k : continuation) (vs : list value) (rs' : region_state) :
  end_region rs x kind ended_by region body <> Region_exit k vs rs'.
Proof.
  unfold end_region. destruct (kind_eqb kind Value); [ | discriminate ].
  destruct (eval_simple (state_vars (rs_state rs)) region)
    as [[ | | | m | | | | | ] | ]; try discriminate.
  destruct (cut m (rs_frames rs)) as [[above below] | ]; discriminate.
Qed.

Lemma end_region_kind (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  end_region rs x kind ended_by region body = Region_step rs' -> kind = Value.
Proof.
  unfold end_region. destruct (kind_eqb kind Value) eqn:Hkind;
    [ | discriminate ].
  intros _. exact (proj1 (kind_eqb_eq kind Value) Hkind).
Qed.

(* The erased term binds the unit value: one step. The frames that
   are cut bound symbols that no value of the state after may be
   evaluated with, so what was related is related. *)
Lemma end_region_erased (rs : region_state) (s' : state) (x : variable)
    (kind : flambda_kind) (d d' : named) (ended_by : mark_kind)
    (region : simple) (body : expr) :
  symbols_fresh rs = true ->
  values_live rs = true ->
  locals_once rs ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Let x kind d body ->
  (∀ rconts : region_conts,
    erase rconts (Let x kind d body) = Let x kind d' (erase rconts body)) ->
  (∀ (statics : static_env) (vars : env),
    eval_named statics vars d' = Some unit_value) ->
  sim_result rs s' (end_region rs x kind ended_by region body).
Proof.
  intros Hfresh Hlive Honce
    (rconts & He & Hcalls & Hagree & Hvars & Hconts & Hstatics & Hheap)
    Hexpr Herase Hunit.
  destruct (end_region rs x kind ended_by region body)
    as [rs1 | k vs rs1 | fault] eqn:Hend;
    [ | exfalso; exact (end_region_no_exit _ _ _ _ _ _ _ _ _ Hend)
    | exact I ].
  pose proof (end_region_kind _ _ _ _ _ _ _ Hend) as ->.
  destruct (end_region_step _ _ _ _ _ _ _ Hend)
    as (m & above & f & below & counts & _ & Hframes & ->).
  right. left.
  destruct s' as [e' vars' conts' statics' hp'].
  cbn [state_expr state_vars state_conts state_statics state_heap] in *.
  rewrite Hexpr, Herase in He. subst e'.
  rewrite step_let, Hunit.
  eexists. split; [ reflexivity | ].
  pose proof (proj1 (values_live_parts rs) Hlive) as (_ & _ & L3 & L4).
  pose
    (rs1 :=
       bind_let rs x unit_value body (state_statics (rs_state rs)) below
         (rs_next rs) counts).
  exists rconts.
  split; [ reflexivity | ].
  split; [ rewrite Hexpr in Hcalls; exact Hcalls | ].
  split; [ exact Hagree | ].
  split.
  { apply
      (env_erased_cons rs1 x unit_value unit_value (state_vars (rs_state rs))
         vars');
      [ intros _; apply Erased_const | ].
    exact
      (env_erased_cut rs rs1 above f Hfresh Hlive Honce eq_refl eq_refl
         Hframes _ _ L3 Hvars). }
  split;
    [ exact
        (conts_erased_cut rs rs1 above f Hfresh Hlive Honce eq_refl eq_refl
           Hframes _ _ L4 Hconts)
    | ].
  split;
    [ exact
        (statics_erased_cut rs rs1 above f Hfresh Hlive Honce eq_refl eq_refl
           Hframes _ Hstatics)
    | exact Hheap ].
Qed.

(* ======================================================================== *)
(* A [Let] that is not of a primitive of regions                            *)
(* ======================================================================== *)

Lemma named_region_primitive_dec (d : named) :
  (∃ (p : region_primitive) (args : list simple),
    d = Prim (Region_primitive p) args)
  ∨ (∀ (p : region_primitive) (args : list simple),
      d <> Prim (Region_primitive p) args).
Proof.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | project_from slot closure | move_from move_to closure ];
    try (right; intros p0 args0; discriminate).
  destruct p as [ | | | | rp]; try (right; intros p0 args0; discriminate).
  left. exists rp, args. reflexivity.
Qed.

Lemma erase_let_plain (rconts : region_conts) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  (∀ (p : region_primitive) (args : list simple),
    d <> Prim (Region_primitive p) args) ->
  erase rconts (Let x kind d body) = Let x kind d (erase rconts body).
Proof.
  intros Hnot.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | project_from slot closure | move_from move_to closure ];
    try reflexivity.
  destruct p as [ | | | | rp]; try reflexivity.
  contradiction (Hnot rp args eq_refl).
Qed.

Lemma head_plain_let_plain (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  (∀ (p : region_primitive) (args : list simple),
    d <> Prim (Region_primitive p) args) ->
  head_plain (Let x kind d body)
  = match marker_of x body with
    | No_marker => true
    | Marker _ _ _ | Bad_marker => false
    end.
Proof.
  intros Hnot.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | project_from slot closure | move_from move_to closure ];
    try reflexivity.
  destruct p as [ | | | | rp]; try reflexivity.
  contradiction (Hnot rp args eq_refl).
Qed.

(* The step of such a [Let]: the test of its simples, none of which is
   at the place of a region, and then what follows it says which step
   it is. *)
Lemma region_step_let (rs : region_state) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  state_expr (rs_state rs) = Let x kind d body ->
  (∀ (p : region_primitive) (args : list simple),
    d <> Prim (Region_primitive p) args) ->
  region_step rs
  = match simples_fault rs (simples_of_named d) [] with
    | Some fault => Region_undefined fault
    | None =>
      match marker_of x body with
      | Marker y region body' =>
        local_allocation (count_uses rs) x y kind d region body'
      | Bad_marker => Region_undefined Not_of_the_conversion
      | No_marker =>
        if is_allocation d
           && existsb (simple_local (count_uses rs)) (simples_of_named d)
        then Region_undefined Heap_holds_local
        else plain_step (count_uses rs)
      end
    end.
Proof.
  intros Hexpr Hnot. unfold region_step, region_places. rewrite Hexpr.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | project_from slot closure | move_from move_to closure ];
    try reflexivity.
  destruct p as [ | | | | rp]; try reflexivity.
  contradiction (Hnot rp args eq_refl).
Qed.

(* The step of a [Let] of a primitive of regions. *)
Lemma region_step_region_primitive (rs : region_state) (x : variable)
    (kind : flambda_kind) (p : region_primitive) (args : list simple)
    (body : expr) :
  state_expr (rs_state rs)
  = Let x kind (Prim (Region_primitive p) args) body ->
  region_step rs
  = match p, args with
    | Begin_region, [] => begin_region (count_uses rs) x kind Plain_mark body
    | Begin_try_region, [] =>
      begin_region (count_uses rs) x kind Try_mark body
    | End_region, [region] =>
      end_region (count_uses rs) x kind Plain_mark region body
    | End_try_region, [region] =>
      end_region (count_uses rs) x kind Try_mark region body
    | _, _ => Region_undefined Not_of_the_conversion
    end.
Proof. intros Hexpr. unfold region_step. rewrite Hexpr. reflexivity. Qed.

(* ======================================================================== *)
(* A local allocation                                                       *)
(* ======================================================================== *)

Lemma lookup_nodup_in {A : Type} (name : positive) (a : A)
    (bindings : list (positive * A)) :
  NoDup (map fst bindings) ->
  In (name, a) bindings ->
  lookup name bindings = Some a.
Proof.
  induction bindings as [ | [bound b] bindings IH]; [ intros _ [] | ].
  simpl. intros Hnodup [Heq | Hin].
  - injection Heq as -> ->. rewrite Pos.eqb_refl. reflexivity.
  - inversion Hnodup as [ | k l Hnotin Hrest]; subst.
    destruct (Pos.eqb name bound) eqn:Hname.
    + apply Pos.eqb_eq in Hname. subst bound. exfalso. apply Hnotin.
      apply in_map_iff. exists (name, a). split; [ reflexivity | exact Hin ].
    + exact (IH Hrest Hin).
Qed.

Lemma number_slots_fst (function_decls : list (function_slot * code_id)) :
  ∀ next : symbol,
    map fst (number_slots function_decls next) = map fst function_decls.
Proof.
  induction function_decls as [ | [slot code] function_decls IH];
    intros next; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(* What a local allocation makes is of the kind [Value]. *)
Lemma made_kind (next : symbol) (v : value) (sym : symbol)
    (bindings : list (symbol * statically_allocated_value)) :
  localise next v = Some (sym, bindings) -> kind_of_value v = Value.
Proof. destruct v; simpl; intros H; try discriminate H; reflexivity. Qed.

(* The symbol of a local allocation is erased to what the value that
   was allocated is erased to, in a state that binds what [localise]
   gives. *)
Lemma localise_erased (rs1 : region_state) (next : symbol) (v v' : value)
    (sym : symbol) (bindings : list (symbol * statically_allocated_value)) :
  localise next v = Some (sym, bindings) ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) bindings ->
    drawn rs1 s = true
    ∧ lookup s (static_values (live_statics rs1)) = Some allocated) ->
  value_erased rs1 v v' ->
  value_erased rs1 (Value_symbol sym) v'.
Proof.
  intros Hloc Hbound Hv.
  destruct v
    as [ c | code slots | slot decls slots | s | tag fields | elements | c
       | l | callee args args_arity ];
    simpl in Hloc; try discriminate.
  - injection Hloc as <- <-.
    destruct (Hbound next (Static_closure code slots) (or_introl eq_refl))
      as [Hdrawn Hl].
    inversion Hv; subst.
    match goal with
    | H : Forall2 _ slots ?slots' |- _ =>
      exact (Erased_local_closure rs1 next code slots slots' Hdrawn Hl H)
    end.
  - destruct (lookup slot (number_slots decls next)) as [s0 | ] eqn:Hslot;
      [ | discriminate ].
    simpl in Hloc. injection Hloc as <- <-.
    inversion Hv; subst.
    match goal with
    | H : Forall2 _ slots ?slots' |- _ =>
      apply
        (Erased_local_closure_of_set rs1 s0 slot (number_slots decls next)
           decls slots slots');
        [ | exact Hslot | exact (number_slots_fst decls next) | | exact H ]
    end.
    + refine (proj1 (Hbound s0 _ _)).
      apply in_map_iff. exists (slot, s0).
      split; [ reflexivity | exact (lookup_in _ _ _ Hslot) ].
    + intros slot0 s1 Hin. refine (proj2 (Hbound s1 _ _)).
      apply in_map_iff. exists (slot0, s1). split; [ reflexivity | exact Hin ].
  - injection Hloc as <- <-.
    destruct (Hbound next (Static_block tag fields) (or_introl eq_refl))
      as [Hdrawn Hl].
    inversion Hv; subst.
    match goal with
    | H : Forall2 _ fields ?fields' |- _ =>
      exact (Erased_local_block rs1 next tag fields fields' Hdrawn Hl H)
    end.
  - injection Hloc as <- <-.
    destruct (Hbound next (Static_immutable_array elements) (or_introl eq_refl))
      as [Hdrawn Hl].
    inversion Hv; subst.
    match goal with
    | H : Forall2 _ elements ?elements' |- _ =>
      exact (Erased_local_array rs1 next elements elements' Hdrawn Hl H)
    end.
  - injection Hloc as <- <-.
    destruct
      (Hbound next (Static_boxed_number (Value_const c)) (or_introl eq_refl))
      as [Hdrawn Hl].
    inversion Hv; subst.
    exact (Erased_local_boxed_number rs1 next c Hdrawn Hl).
Qed.

Lemma env_erased_shadow (rs : region_state) (x : variable) (v : value)
    (vars vars' : env) :
  env_erased rs ((x, v) :: vars) vars' ->
  env_erased rs ((x, v) :: (x, v) :: vars) vars'.
Proof.
  intros H z w Hz Hu. apply (H z w); [ | exact Hu ].
  simpl in Hz |- *. destruct (Pos.eqb z x); exact Hz.
Qed.

(* The erased term binds the value on the heap, and then the variable
   of the [Local_allocation] to the first, unless they are one
   variable: one step, or two. The value that was allocated is erased
   to what the second state binds; the symbol that the first state
   binds is bound to it in the state after, so the symbol is erased to
   the same. *)
Lemma local_allocation_erased (rs : region_state) (s' : state)
    (x y : variable) (kind : flambda_kind) (d : named)
    (region : option simple) (body body' : expr) :
  symbols_fresh rs = true ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Let x kind d body ->
  (∀ (p : region_primitive) (args : list simple),
    d <> Prim (Region_primitive p) args) ->
  marker_of x body = Marker y region body' ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  sim_result rs s' (local_allocation rs x y kind d region body').
Proof.
  intros Hfresh
    (rconts & He & Hcalls & Hagree & Hvars & Hconts & Hstatics & Hheap)
    Hexpr Hnot Hmarker Husable.
  destruct (local_allocation rs x y kind d region body')
    as [rs1 | k vs rs1 | fault] eqn:Hlocal;
    [ | exfalso; exact (local_allocation_no_exit _ _ _ _ _ _ _ _ _ _ Hlocal)
    | exact I ].
  destruct (local_allocation_step _ _ _ _ _ _ _ _ Hlocal)
    as (v & sym & bindings & Halloc & Hv & Hloc & Hexpr1 & Hvars1 & Hconts1
        & Hbase1 & Hnext1 & Hcode1 & Hframes1).
  destruct (local_allocation_kind_heap _ _ _ _ _ _ _ _ Hlocal)
    as [Hkind Hheap1].
  apply kind_eqb_eq in Hkind. subst kind.
  destruct (marker_of_marker_var _ _ _ _ _ Hmarker)
    as (kind' & rest & -> & Hkind').
  apply kind_eqb_eq in Hkind'. subst kind'.
  (* What the invariant gives. *)
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & [Hheld_vars _] & _).
  destruct (symbols_fresh_tests rs Hfresh) as (F1 & F2 & F4 & F5 & _).
  pose proof F2 as Hd_below. rewrite Hexpr in Hd_below. simpl in Hd_below.
  apply andb_true_iff in Hd_below as [Hd_below _].
  pose proof (proj1 (proj2 (proj2 Hstatics))) as Hsymbols.
  (* The value, erased. *)
  destruct
    (eval_named_erased rs (state_statics s') _ _ d v Hsymbols Hvars Hd_below
       (simples_usable_of_test rs _ Husable) Hv)
    as (v' & Hv' & Hrelated).
  pose proof (kind_of_value_erased rs v v' Hrelated) as Hvkind'.
  rewrite (made_kind _ _ _ _ Hloc) in Hvkind'.
  (* The value names no symbol at or above the counter. *)
  assert (Hv_below : value_below (rs_next rs) v = true).
  { destruct (eval_named_makes _ _ d v Halloc Hv) as [Hmade Hparts].
    apply (value_below_made _ v Hmade). apply forallb_forall. intros w Hw.
    destruct (Hparts w Hw) as (sim & Hsim & Hw_eval).
    rewrite Hexpr in F2.
    exact
      (named_simples_below (rs_base rs) (rs_next rs) _ x Value d _ sim w F1 F2
         Hheld_vars Hsim Hw_eval). }
  (* What the state after binds. *)
  destruct (localise_makes _ _ _ _ Hloc) as (_ & _ & Hbindings).
  assert (Hvalues :
    static_values (live_statics rs1)
    = bindings ++ static_values (live_statics rs)).
  { rewrite !live_statics_values.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      destruct Hframes1 as [Hf Hs]; rewrite Hf, Hs;
      [ reflexivity | simpl; rewrite <- !app_assoc; reflexivity ]. }
  assert (Hgrow :
    ∀ s : symbol,
      lookup s (static_values (state_statics (rs_state rs))) <> None ->
      lookup s (static_values (state_statics (rs_state rs1))) <> None).
  { destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      destruct Hframes1 as [_ Hs]; rewrite Hs; [ | auto ].
    intros s Hs0. rewrite lookup_app.
    destruct (lookup s bindings); [ discriminate | exact Hs0 ]. }
  assert (Hfrom :
    ∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) bindings -> (rs_next rs <= s)%positive)
    by (intros s allocated Hin; exact (proj1 (Hbindings s allocated Hin))).
  pose proof
    (value_erased_allocated rs rs1 bindings Hfresh Hbase1 Hvalues Hfrom v v'
       Hrelated Hv_below)
    as Hrelated1.
  assert (Hsym : value_erased rs1 (Value_symbol sym) v').
  { apply (localise_erased rs1 (rs_next rs) v v' sym bindings Hloc);
      [ | exact Hrelated1 ].
    intros s allocated Hin. split.
    - unfold drawn. rewrite Hbase1. apply Pos.leb_le.
      pose proof (Hfrom s allocated Hin). lia.
    - rewrite Hvalues, lookup_app,
        (lookup_nodup_in s allocated bindings (localise_once _ _ _ _ Hloc)
           Hin).
      reflexivity. }
  pose proof
    (env_erased_allocated rs rs1 bindings Hfresh Hbase1 Hvalues Hfrom _ _ F4
       Hvars)
    as Hvars_rs1.
  pose proof
    (conts_erased_allocated rs rs1 bindings Hfresh Hbase1 Hvalues Hfrom _ _
       F5 Hconts)
    as Hconts_rs1.
  pose proof
    (statics_erased_allocated rs rs1 bindings Hfresh Hbase1 Hvalues Hcode1
       Hgrow Hfrom _ Hstatics)
    as Hstatics_rs1.
  (* The steps of the second state. *)
  destruct s' as [e' vars' conts' statics' hp'].
  cbn [state_expr state_vars state_conts state_statics state_heap] in *.
  rewrite Hexpr, (erase_let_plain rconts x Value d _ Hnot) in He.
  change
    (erase rconts
       (Let y Value (Prim (Region_primitive Local_allocation) (Var x :: rest))
          body'))
    with
      (match (if Pos.eqb y x then None else Some (Simple (Var x))) with
       | None => erase rconts body'
       | Some d0 => Let y Value d0 (erase rconts body')
       end)
    in He.
  subst e'.
  assert (Hcalls1 : calls_return_no_region rconts body' = true)
    by (rewrite Hexpr in Hcalls; exact Hcalls).
  destruct (Pos.eqb y x) eqn:Hyx.
  - (* The second [Let] binds the variable of the first: one step. *)
    apply Pos.eqb_eq in Hyx. subst y.
    right. left. rewrite step_let, Hv', Hvkind'.
    eexists. split; [ reflexivity | ].
    exists rconts. rewrite Hexpr1, Hvars1, Hconts1, Hheap1.
    cbn [state_expr state_vars state_conts state_statics state_heap].
    split; [ reflexivity | ]. split; [ exact Hcalls1 | ].
    split; [ exact Hagree | ].
    split.
    { apply env_erased_shadow.
      apply env_erased_cons; [ intros _; exact Hsym | exact Hvars_rs1 ]. }
    split; [ exact Hconts_rs1 | ].
    split; [ exact Hstatics_rs1 | exact Hheap ].
  - (* It binds another: two steps. *)
    right. right. rewrite step_let, Hv', Hvkind'.
    eexists. eexists. split; [ reflexivity | ].
    rewrite step_let. cbn [eval_named eval_simple lookup].
    rewrite Pos.eqb_refl, Hvkind'.
    split; [ reflexivity | ].
    exists rconts. rewrite Hexpr1, Hvars1, Hconts1, Hheap1.
    cbn [state_expr state_vars state_conts state_statics state_heap].
    split; [ reflexivity | ]. split; [ exact Hcalls1 | ].
    split; [ exact Hagree | ].
    split.
    { apply env_erased_cons; [ intros _; exact Hsym | ].
      apply env_erased_cons; [ intros _; exact Hsym | exact Hvars_rs1 ]. }
    split; [ exact Hconts_rs1 | ].
    split; [ exact Hstatics_rs1 | exact Hheap ].
Qed.

(* ======================================================================== *)
(* A region given to a continuation                                         *)
(* ======================================================================== *)

Lemma simple_usable_value (rs : region_state) (s : simple) (v : value) :
  simple_usable rs s = true ->
  eval_simple (state_vars (rs_state rs)) s = Some v ->
  value_usable rs v = true.
Proof. unfold simple_usable. intros H Hv. rewrite Hv in H. exact H. Qed.

(* A mark that is on the stack may not be evaluated. *)
Lemma open_mark_not_usable (rs : region_state) (s : simple) :
  simple_open_mark rs s = true -> simple_usable rs s = false.
Proof.
  unfold simple_open_mark, simple_usable.
  destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ];
    [ | discriminate ].
  unfold value_open_mark.
  destruct v as [ | | | m | | | | | ]; try discriminate.
  intros H. apply andb_true_iff in H as [H _]. apply negb_true_iff in H.
  exact H.
Qed.

(* Where the test of the simples passes and one of them is a mark that
   is on the stack, one of the places is that of a region. *)
Lemma simples_fault_mark_place (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    simples_fault rs simples places = None ->
    existsb (simple_open_mark rs) simples = true ->
    existsb (fun place => place) places = true.
Proof.
  induction simples as [ | s simples IH]; intros places Hfault Hmark;
    [ discriminate | ].
  simpl in Hfault, Hmark.
  destruct (simple_usable rs s) eqn:Husable.
  - destruct (simple_open_mark rs s) eqn:Hm;
      [ rewrite (open_mark_not_usable rs s Hm) in Husable; discriminate | ].
    simpl in Hmark. pose proof (IH (tl places) Hfault Hmark) as Hrest.
    destruct places as [ | place places]; simpl in *;
      [ exact Hrest | rewrite Hrest; apply orb_true_r ].
  - destruct (simple_open_mark rs s); [ | discriminate ].
    destruct places as [ | [ | ] places]; simpl in Hfault; try discriminate.
    reflexivity.
Qed.

(* The arguments that are not at the place of a region have values in
   the second state: the values that they have in the first,
   erased. *)
Lemma eval_simples_dropped (rs : region_state) (vars' : env)
    (args : list simple) :
  env_erased rs (state_vars (rs_state rs)) vars' ->
  ∀ (places : list bool) (vs : list value),
    forallb (simple_below (rs_base rs)) args = true ->
    simples_fault rs args places = None ->
    eval_simples (state_vars (rs_state rs)) args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (drop_regions places args) = Some vs'
      ∧ Forall2 (value_erased rs) (drop_regions places vs) vs'.
Proof.
  intros Hvars. induction args as [ | s args IH];
    intros places vs Hbelow Hfault Hvs.
  - simpl in Hvs. injection Hvs as <-. exists [].
    destruct places; (split; [ reflexivity | constructor ]).
  - simpl in Hbelow. apply andb_true_iff in Hbelow as [Hs_below Hbelow].
    simpl in Hvs.
    destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ] eqn:Hs;
      [ | discriminate ].
    destruct (eval_simples (state_vars (rs_state rs)) args) as [rest | ]
      eqn:Hrest; [ | discriminate ].
    injection Hvs as <-.
    simpl in Hfault.
    assert (Hkeep :
      simple_usable rs s = true ->
      simples_fault rs args (tl places) = None ->
      ∃ vs' : list value,
        eval_simples vars' (s :: drop_regions (tl places) args) = Some vs'
        ∧ Forall2 (value_erased rs) (v :: drop_regions (tl places) rest) vs').
    { intros Husable Hfault'.
      destruct
        (eval_simple_erased rs _ vars' s v Hvars Hs_below Hs
           (simple_usable_value rs s v Husable Hs))
        as (v' & Hv' & Hv).
      destruct (IH (tl places) rest Hbelow Hfault' eq_refl)
        as (rest' & Hrest' & Hrelated).
      exists (v' :: rest'). simpl. rewrite Hv', Hrest'.
      split; [ reflexivity | constructor; assumption ]. }
    destruct places as [ | place places]; simpl in Hfault, Hkeep.
    + destruct (simple_usable rs s) eqn:Husable;
        [ exact (Hkeep eq_refl Hfault) | ].
      destruct (simple_open_mark rs s); discriminate.
    + destruct place.
      * apply (IH places rest Hbelow); [ | reflexivity ].
        destruct (simple_usable rs s); [ exact Hfault | ].
        destruct (simple_open_mark rs s); [ exact Hfault | discriminate ].
      * destruct (simple_usable rs s) eqn:Husable;
          [ exact (Hkeep eq_refl Hfault) | ].
        destruct (simple_open_mark rs s); discriminate.
Qed.

Lemma region_kind_not_mark (rs : region_state) (v : value) :
  region_kind_of_value rs v <> Region ->
  region_kind_of_value rs v = kind_of_value v.
Proof.
  unfold region_kind_of_value.
  destruct (value_open_mark rs v);
    [ intros H; contradiction (H eq_refl) | reflexivity ].
Qed.

Lemma region_kind_mark (rs : region_state) (v : value) :
  region_kind_of_value rs v = Region ->
  kind_of_value v <> Region ->
  value_usable rs v = false.
Proof.
  unfold region_kind_of_value.
  destruct (value_open_mark rs v) eqn:Hm;
    [ intros _ _ | intros H Hn; contradiction (Hn H) ].
  unfold value_open_mark in Hm.
  destruct v as [ | | | m | | | | | ]; try discriminate.
  apply andb_true_iff in Hm as [Hm _]. apply negb_true_iff in Hm. exact Hm.
Qed.

(* The places of the regions among parameters of the kinds given. *)
Abbreviation places_of kinds :=
  (map (fun kind => kind_eqb kind Region) kinds).

(* The values that are not given to a region parameter are of the
   kinds of the parameters that are left. *)
Lemma dropped_kinds (rs : region_state) (vs : list value) :
  map kind_of_value
    (drop_regions (places_of (map (region_kind_of_value rs) vs)) vs)
  = drop_regions (places_of (map (region_kind_of_value rs) vs))
      (map (region_kind_of_value rs) vs).
Proof.
  induction vs as [ | v vs IH]; [ reflexivity | ].
  simpl. destruct (kind_eqb (region_kind_of_value rs v) Region) eqn:Hkind;
    [ exact IH | ].
  simpl. rewrite IH. f_equal. symmetry. apply region_kind_not_mark.
  intros H. rewrite H, kind_eqb_refl in Hkind. discriminate.
Qed.

(* The values at the places of regions may not be evaluated. *)
Fixpoint regions_dead (rs : region_state) (places : list bool)
    (vs : list value) : Prop :=
  match places, vs with
  | place :: places, v :: vs =>
    (place = true -> value_usable rs v = false) ∧ regions_dead rs places vs
  | _, _ => True
  end.

(* A value given to a region parameter is a mark, where no value is of
   the kind [Region]. *)
Lemma region_values_dead (rs : region_state) (vs : list value) :
  Forall (fun v => kind_of_value v <> Region) vs ->
  regions_dead rs (places_of (map (region_kind_of_value rs) vs)) vs.
Proof.
  induction 1 as [ | v vs Hv _ IH]; simpl; [ exact I | ].
  split; [ | exact IH ].
  intros Hplace. apply kind_eqb_eq in Hplace.
  exact (region_kind_mark rs v Hplace Hv).
Qed.

(* The parameters that are not regions are bound in the second state to
   the values erased; a region parameter is bound in the first alone,
   to a mark. *)
Lemma bind_params_dropped (rs : region_state) :
  ∀ (places : list bool) (params : list variable) (vs vs' : list value)
    (vars vars' bound : env),
    regions_dead rs places vs ->
    Forall2 (value_erased rs) (drop_regions places vs) vs' ->
    env_erased rs vars vars' ->
    bind_params params vs vars = Some bound ->
    ∃ bound' : env,
      bind_params (drop_regions places params) vs' vars' = Some bound'
      ∧ env_erased rs bound bound'.
Proof.
  induction places as [ | place places IH];
    intros params vs vs' vars vars' bound Hdead Hvs Hvars Hbind.
  - exact
      (bind_params_erased rs params vs vs' vars vars' bound
         (values_erased_if_usable_intro rs _ _ Hvs) Hvars Hbind).
  - destruct params as [ | x params]; destruct vs as [ | v vs];
      simpl in Hbind; try discriminate.
    + injection Hbind as <-. simpl in Hvs. inversion Hvs; subst.
      exists vars'. split; [ reflexivity | exact Hvars ].
    + simpl in Hdead. destruct Hdead as [Hplace Hdead].
      destruct place; simpl in Hvs |- *.
      * exact
          (IH params vs vs' _ vars' bound Hdead Hvs
             (env_erased_cons_dead rs x v vars vars' (Hplace eq_refl) Hvars)
             Hbind).
      * inversion Hvs as [ | v0 v' rest rest' Hv Hrest]; subst. simpl.
        exact
          (IH params vs rest' _ _ bound Hdead Hrest
             (env_erased_cons rs x v v' vars vars' (fun _ => Hv) Hvars)
             Hbind).
Qed.

(* No value of an argument is of the kind [Region], in a state that has
   no poison constant of that kind. *)
Lemma args_not_region (rs : region_state) (args : list simple)
    (vs : list value) :
  no_poison rs = true ->
  (∀ sim : simple, In sim args -> simple_no_region_poison sim = true) ->
  eval_simples (state_vars (rs_state rs)) args = Some vs ->
  Forall (fun v => kind_of_value v <> Region) vs.
Proof.
  intros Hpoison Hargs Hvs.
  destruct (proj1 (no_poison_parts rs) Hpoison) as (_ & [Hheld _] & _).
  apply (eval_simples_forall _ (state_vars (rs_state rs)) args vs);
    [ | exact Hvs ].
  intros sim w Hin Hw. apply value_no_region_poison_kind.
  exact (eval_simple_no_region_poison _ sim w (Hargs sim Hin) Hheld Hw).
Qed.

(* The erased term gives the continuation the arguments that are not
   regions, and the erased handler binds them: one step. The
   parameters that are regions are bound in the first state alone, to
   marks. *)
Lemma region_apply_cont_erased (rs : region_state) (s' : state)
    (ac : apply_cont) :
  symbols_fresh rs = true ->
  no_poison rs = true ->
  state_erased rs s' ->
  state_expr (rs_state rs) = Apply_cont ac ->
  simples_fault rs (ac_args ac) (region_places rs) = None ->
  existsb (simple_open_mark rs) (ac_args ac) = true ->
  sim_result rs s' (region_apply_cont rs ac).
Proof.
  intros Hfresh Hpoison
    (rconts & He & Hcalls & Hagree & Hvars & Hconts & Hstatics & Hheap)
    Hexpr Hfault Hmark.
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  rewrite Hexpr in Hbelow.
  change (forallb (simple_below (rs_base rs)) (ac_args ac) = true) in Hbelow.
  assert (Hargs :
    ∀ sim : simple,
      In sim (ac_args ac) -> simple_no_region_poison sim = true).
  { intros sim Hin.
    destruct (proj1 (no_poison_parts rs) Hpoison) as ([Hwritten _] & _).
    rewrite Hexpr in Hwritten.
    exact (evaluated_simples_no_region_poison (Apply_cont ac) sim Hwritten Hin).
  }
  pose proof (resolve_erased rs (ac_continuation ac) _ _ Hconts) as Hc.
  unfold region_places in Hfault. rewrite Hexpr in Hfault.
  unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [ h def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ]
    eqn:Hresolve;
    try (exfalso;
         pose proof (simples_fault_mark_place rs _ [] Hfault Hmark) as H;
         discriminate H).
  destruct h as [params [kinds is_exn] handler].
  unfold region_params, handler_kinds in Hfault.
  cbn [hm_params_kinds] in Hfault.
  cbn [hm_params_kinds].
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ]
    eqn:Hvs; [ | exact I ].
  pose proof (args_not_region rs (ac_args ac) vs Hpoison Hargs Hvs)
    as Hnot_region.
  destruct (kinds_eqb (map (region_kind_of_value rs) vs) kinds) eqn:Hkinds;
    [ | exact I ].
  apply kinds_eqb_eq in Hkinds. subst kinds.
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | exact I ].
  right. left.
  (* The continuation is one that [erase] lists. *)
  pose proof (Hagree (ac_continuation ac)) as Hk.
  destruct (lookup (ac_continuation ac) rconts) as [places | ] eqn:Hlisted.
  2: { exfalso. rewrite Hresolve in Hk.
       unfold cont_takes_no_region, handler_takes_no_region, region_params,
         handler_kinds in Hk.
       cbn [hm_params_kinds] in Hk.
       rewrite (simples_fault_mark_place rs _ _ Hfault Hmark) in Hk.
       discriminate Hk. }
  destruct Hk as (h0 & vars0 & conts0 & recursive0 & Hlookup & ->).
  unfold resolve in Hresolve. rewrite Hlookup in Hresolve.
  injection Hresolve as -> -> -> ->.
  unfold region_params, handler_kinds in Hlisted.
  cbn [hm_params_kinds] in Hlisted.
  (* The closure of the second state. *)
  inversion Hc
    as [
       | rconts0 h1 vars1 vars1' conts1 conts1' recursive1 Hh Hrecursive
           Hagree0 Hvars0 Hconts0
       |
       | ].
  subst h1 vars1 conts1 recursive1.
  match goal with
  | H : Closure _ _ _ _ = resolve _ _ |- _ => rename H into Hc'
  end.
  destruct s' as [e' vars' conts' statics' hp'].
  cbn [state_expr state_vars state_conts state_statics state_heap] in *.
  rewrite Hexpr in He. cbn [erase] in He. unfold erase_apply_cont in He.
  rewrite Hlisted in He. subst e' hp'.
  rewrite step_apply_cont. cbn [ac_args ac_continuation].
  destruct
    (eval_simples_dropped rs vars' (ac_args ac) Hvars _ vs Hbelow Hfault Hvs)
    as (vs' & Hvs' & Hrelated).
  rewrite Hvs'. unfold apply_continuation. rewrite <- Hc'.
  cbn [erase_handler apply_cont_value hm_params_kinds].
  rewrite (kinds_of_values_erased rs _ vs' Hrelated), dropped_kinds,
    kinds_eqb_refl.
  destruct
    (bind_params_dropped rs _ params vs vs' def_vars vars1' handler_vars
       (region_values_dead rs vs Hnot_region) Hrelated Hvars0 Hbind)
    as (bound' & Hbound' & Hbound).
  rewrite Hbound'.
  eexists. split; [ reflexivity | ].
  exact
    (state_erased_plain_next rs (region_conts_of recursive ++ rconts0) handler
       handler_vars bound' _ _ (state_statics (rs_state rs)) statics'
       (state_heap (rs_state rs)) Hh
       (conts_agree_rec rconts0 def_conts def_vars def_conts recursive
          recursive Hagree0)
       Hbound
       (rec_closures_erased rs rconts0 def_vars vars1' def_conts conts1'
          recursive Hrecursive Hagree0 Hvars0 Hconts0)
       Hstatics).
Qed.

(* The step of an [Apply_cont] one of whose arguments is a mark that is
   on the stack. *)
Lemma region_step_apply_cont (rs : region_state) (ac : apply_cont) :
  state_expr (rs_state rs) = Apply_cont ac ->
  existsb (simple_open_mark rs) (ac_args ac) = true ->
  region_step rs
  = match simples_fault rs (ac_args ac) (region_places rs) with
    | Some fault => Region_undefined fault
    | None => region_apply_cont (count_uses rs) ac
    end.
Proof.
  intros Hexpr Hmark. unfold region_step. rewrite Hexpr.
  cbn [evaluated_simples].
  destruct (simples_fault rs (ac_args ac) (region_places rs));
    [ reflexivity | ].
  destruct (count_uses_with_counts rs) as [counts Hcounts].
  rewrite Hcounts, existsb_open_mark_with_counts, Hmark. reflexivity.
Qed.

(* ======================================================================== *)
(* The counts change nothing that the tests read                            *)
(* ======================================================================== *)

#[local] Transparent live_statics.

Lemma symbols_fresh_with_counts (rs : region_state) (counts : region_counts) :
  symbols_fresh (with_counts rs counts) = symbols_fresh rs.
Proof. reflexivity. Qed.

Lemma values_live_with_counts (rs : region_state) (counts : region_counts) :
  values_live rs = true -> values_live (with_counts rs counts) = true.
Proof.
  intros Hlive.
  apply (values_live_transfer rs (with_counts rs counts)); try reflexivity;
    [ exact Hlive | ].
  apply values_live_parts in Hlive as (_ & _ & L3 & L4).
  exact (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)).
Qed.

Lemma no_poison_with_counts (rs : region_state) (counts : region_counts) :
  no_poison (with_counts rs counts) = no_poison rs.
Proof. reflexivity. Qed.

Lemma simples_fault_with_counts (rs : region_state) (counts : region_counts)
    (simples : list simple) :
  ∀ places : list bool,
    simples_fault (with_counts rs counts) simples places
    = simples_fault rs simples places.
Proof.
  induction simples as [ | s simples IH]; intros places; [ reflexivity | ].
  simpl. rewrite !IH. reflexivity.
Qed.

#[local] Opaque live_statics.

Lemma sim_result_with_counts (rs : region_state) (counts : region_counts)
    (s' : state) (result : region_step_result) :
  sim_result (with_counts rs counts) s' result -> sim_result rs s' result.
Proof. destruct result; exact (fun H => H). Qed.

(* ======================================================================== *)
(* A step of the machine of regions                                         *)
(* ======================================================================== *)

(* Where the machine of regions does what [step] does. *)
Lemma region_step_shared_sim (rs : region_state) (s' : state) :
  state_erased rs s' ->
  symbols_fresh rs = true ->
  no_poison rs = true ->
  head_plain (state_expr (rs_state rs)) = true ->
  (∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    existsb (simple_open_mark rs) (ac_args ac) = false) ->
  sim_result rs s' (region_step rs).
Proof.
  intros Herased Hfresh Hpoison Hhead Hno_mark.
  apply result_erased_sim.
  apply region_step_shared;
    [ exact Herased | exact Hhead | exact Hno_mark
    | exact (symbols_fresh_expr rs Hfresh) | ].
  intros ac vs Hexpr Hvs.
  apply (args_not_region rs (ac_args ac) vs Hpoison); [ | exact Hvs ].
  intros sim Hin.
  destruct (proj1 (no_poison_parts rs) Hpoison) as ([Hwritten _] & _).
  rewrite Hexpr in Hwritten.
  exact (evaluated_simples_no_region_poison (Apply_cont ac) sim Hwritten Hin).
Qed.

(* One step of the machine of regions, from a state that has the
   invariants, is answered by the second state. *)
Theorem region_step_simulated (rs : region_state) (s' : state) :
  state_erased rs s' ->
  region_invariant rs = true ->
  no_poison rs = true ->
  locals_once rs ->
  sim_result rs s' (region_step rs).
Proof.
  intros Herased Hinv Hpoison Honce.
  apply andb_true_iff in Hinv as [Hfresh Hlive].
  destruct (count_uses_with_counts rs) as [counts Hcounts].
  pose (rsc := with_counts rs counts).
  assert (Herased_c : state_erased rsc s')
    by exact
         (state_erased_same rs rsc s' (same_statics_with_counts rs counts)
            eq_refl Herased).
  assert (Hfresh_c : symbols_fresh rsc = true)
    by (unfold rsc; rewrite symbols_fresh_with_counts; exact Hfresh).
  assert (Hlive_c : values_live rsc = true)
    by exact (values_live_with_counts rs counts Hlive).
  assert (Hpoison_c : no_poison rsc = true)
    by (unfold rsc; rewrite no_poison_with_counts; exact Hpoison).
  assert (Honce_c : locals_once rsc) by exact Honce.
  destruct (state_expr (rs_state rs))
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ]
    eqn:Hexpr.
  - (* A [Let]. *)
    destruct (named_region_primitive_dec d) as [(p & args & ->) | Hnot].
    + (* Of a primitive of regions. *)
      rewrite (region_step_region_primitive rs x kind p args body Hexpr),
        Hcounts.
      destruct p; destruct args as [ | region [ | region' args]];
        try exact I; apply (sim_result_with_counts rs counts).
      * exact
          (begin_region_erased rsc s' x kind _ Plain_mark body Hfresh_c
             Herased_c Hexpr (fun _ => eq_refl)).
      * exact
          (begin_region_erased rsc s' x kind _ Try_mark body Hfresh_c
             Herased_c Hexpr (fun _ => eq_refl)).
      * exact
          (end_region_erased rsc s' x kind _ (Simple unit_simple) Plain_mark
             region body Hfresh_c Hlive_c Honce_c Herased_c Hexpr
             (fun _ => eq_refl) (fun _ _ => eq_refl)).
      * exact
          (end_region_erased rsc s' x kind _
             (Prim (Unary (Opaque_identity Value)) [unit_simple]) Try_mark
             region body Hfresh_c Hlive_c Honce_c Herased_c Hexpr
             (fun _ => eq_refl) (fun _ _ => eq_refl)).
    + destruct (marker_of x body) as [ | y region body' | ] eqn:Hmarker.
      * (* With nothing of regions after it. *)
        apply (region_step_shared_sim rs s' Herased Hfresh Hpoison).
        -- rewrite Hexpr, (head_plain_let_plain x kind d body Hnot), Hmarker.
           reflexivity.
        -- intros ac Heq. rewrite Hexpr in Heq. discriminate Heq.
      * (* With a [Local_allocation] after it. *)
        rewrite (region_step_let rs x kind d body Hexpr Hnot), Hmarker.
        destruct (simples_fault rs (simples_of_named d) []) as [fault | ]
          eqn:Hfault; [ exact I | ].
        rewrite Hcounts. apply (sim_result_with_counts rs counts).
        apply
          (local_allocation_erased rsc s' x y kind d region body body'
             Hfresh_c Herased_c Hexpr Hnot Hmarker).
        apply simples_fault_no_place. unfold rsc.
        rewrite simples_fault_with_counts. exact Hfault.
      * rewrite (region_step_let rs x kind d body Hexpr Hnot), Hmarker.
        destruct (simples_fault rs (simples_of_named d) []); exact I.
  - apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq ].
  - apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq ].
  - apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq ].
  - apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq ].
  - (* An [Apply_cont]. *)
    destruct (existsb (simple_open_mark rs) (ac_args ac)) eqn:Hmark.
    + rewrite (region_step_apply_cont rs ac Hexpr Hmark).
      destruct (simples_fault rs (ac_args ac) (region_places rs))
        as [fault | ] eqn:Hfault; [ exact I | ].
      rewrite Hcounts. apply (sim_result_with_counts rs counts).
      apply
        (region_apply_cont_erased rsc s' ac Hfresh_c Hpoison_c Herased_c
           Hexpr).
      * unfold rsc. rewrite simples_fault_with_counts. exact Hfault.
      * unfold rsc. rewrite existsb_open_mark_with_counts. exact Hmark.
    + apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
        [ rewrite Hexpr; reflexivity | ].
      intros ac0 Heq. rewrite Hexpr in Heq. injection Heq as <-. exact Hmark.
  - apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq ].
  - apply (region_step_shared_sim rs s' Herased Hfresh Hpoison);
      [ rewrite Hexpr; reflexivity
      | intros ac Heq; rewrite Hexpr in Heq; discriminate Heq ].
Qed.

(* ======================================================================== *)
(* Runs                                                                     *)
(* ======================================================================== *)

(* What a run keeps: the relation between the two states, and the three
   tests of the first. *)
Definition related (rs : region_state) (s' : state) : Prop :=
  state_erased rs s'
  ∧ region_invariant rs = true
  ∧ no_poison rs = true
  ∧ locals_once rs.

Lemma related_step (rs rs1 : region_state) (s' s1' : state) :
  related rs s' ->
  region_step rs = Region_step rs1 ->
  state_erased rs1 s1' ->
  related rs1 s1'.
Proof.
  intros (_ & Hinv & Hpoison & Honce) Hstep Herased1.
  split; [ exact Herased1 | ].
  split; [ exact (region_step_keeps_invariant rs rs1 Hinv Hstep) | ].
  split; [ exact (region_step_keeps_no_poison rs rs1 Hpoison Hstep) | ].
  apply andb_true_iff in Hinv as [Hfresh _].
  exact (region_step_keeps_locals_once rs rs1 Hfresh Honce Hstep).
Qed.

Lemma related_sim (rs : region_state) (s' : state) :
  related rs s' -> sim_result rs s' (region_step rs).
Proof.
  intros (H1 & H2 & H3 & H4). exact (region_step_simulated rs s' H1 H2 H3 H4).
Qed.

Lemma region_run_succ (fuel : nat) (rs : region_state) :
  region_run (S fuel) rs
  = match region_step rs with
    | Region_step rs' => region_run fuel rs'
    | result => result
    end.
Proof. reflexivity. Qed.

(* The state that the test tool starts the machine of regions in, and
   the state in which the erased term is run. *)
Lemma related_initial (e : expr) (statics : static_env) (base : symbol) :
  region_names_fresh base e statics = true ->
  erasable e statics = true ->
  related (initial_region_state e statics base)
    (erased_initial_state e statics).
Proof.
  intros Hfresh Herasable. unfold erasable in Herasable.
  apply andb_true_iff in Herasable as [Herasable Hp2].
  apply andb_true_iff in Herasable as [Herasable Hp1].
  apply andb_true_iff in Herasable as [Hr1 Hr2].
  split; [ exact (state_erased_initial e statics base Hfresh Hr1 Hr2) | ].
  split; [ exact (region_invariant_initial e statics base Hfresh) | ].
  split;
    [ exact (no_poison_initial e statics base Hp1 Hp2)
    | exact (locals_once_initial e statics base) ].
Qed.

(* Where the run of the first state exits, that of the second does. *)
Lemma run_exits (fuel : nat) :
  ∀ (rs : region_state) (s' : state) (k : continuation) (vs : list value)
    (rs_end : region_state),
    related rs s' ->
    region_run fuel rs = Region_exit k vs rs_end ->
    ∃ (vs' : list value) (hp : heap),
      behaves s' (Exits k vs' hp)
      ∧ Forall2 (value_erased_if_usable rs_end) vs vs'.
Proof.
  induction fuel as [ | fuel IH]; intros rs s' k vs rs_end Hrelated Hrun;
    [ discriminate | ].
  rewrite region_run_succ in Hrun.
  pose proof (related_sim rs s' Hrelated) as Hsim.
  destruct (region_step rs) as [rs1 | k0 vs0 rs0 | fault] eqn:Hstep;
    [ | | discriminate ].
  - destruct Hsim
      as [ [Herased1 _]
         | [ (s1' & Hstep' & Herased1)
           | (s1' & s2' & Hstep1' & Hstep2' & Herased1) ] ].
    + exact
        (IH rs1 s' k vs rs_end
           (related_step rs rs1 s' s' Hrelated Hstep Herased1) Hrun).
    + destruct
        (IH rs1 s1' k vs rs_end
           (related_step rs rs1 s' s1' Hrelated Hstep Herased1) Hrun)
        as (vs' & hp & [n Hn] & Hvs).
      exists vs', hp. split; [ | exact Hvs ].
      exists (S n). rewrite run_succ, Hstep'. exact Hn.
    + destruct
        (IH rs1 s2' k vs rs_end
           (related_step rs rs1 s' s2' Hrelated Hstep Herased1) Hrun)
        as (vs' & hp & [n Hn] & Hvs).
      exists vs', hp. split; [ | exact Hvs ].
      exists (S (S n)). rewrite run_succ, Hstep1', run_succ, Hstep2'.
      exact Hn.
  - injection Hrun as -> -> ->.
    destruct Hsim as (vs' & hp & Hstep' & Hvs).
    exists vs', hp. split; [ | exact Hvs ].
    exists 1. rewrite run_succ, Hstep'. reflexivity.
Qed.

(* Where the run of the second state exits, that of the first exits or
   stops with no step. The run of the first may make steps that the
   second has none for, the begins of regions: each goes under a
   [Let], so there are no more of them in a row than the expression
   has [Let]s at its head. *)
Lemma run_adds_no_exit (k : continuation) (vs' : list value) (hp : heap) :
  ∀ (n m : nat) (rs : region_state) (s' : state),
    let_depth (state_expr (rs_state rs)) < m ->
    related rs s' ->
    run n s' = Exit k vs' hp ->
    ∃ fuel : nat,
      (∃ fault : region_fault, region_run fuel rs = Region_undefined fault)
      ∨ (∃ (vs : list value) (rs_end : region_state),
          region_run fuel rs = Region_exit k vs rs_end
          ∧ Forall2 (value_erased_if_usable rs_end) vs vs').
Proof.
  induction n as [n IHn] using lt_wf_ind.
  induction m as [ | m IHm]; intros rs s' Hdepth Hrelated Hrun; [ lia | ].
  pose proof (related_sim rs s' Hrelated) as Hsim.
  destruct (region_step rs) as [rs1 | k0 vs0 rs0 | fault] eqn:Hstep.
  - destruct Hsim
      as [ [Herased1 Hless]
         | [ (s1' & Hstep' & Herased1)
           | (s1' & s2' & Hstep1' & Hstep2' & Herased1) ] ].
    + assert (Hdepth1 : let_depth (state_expr (rs_state rs1)) < m) by lia.
      destruct
        (IHm rs1 s' Hdepth1
           (related_step rs rs1 s' s' Hrelated Hstep Herased1) Hrun)
        as [fuel Hfuel].
      exists (S fuel). rewrite region_run_succ, Hstep. exact Hfuel.
    + destruct n as [ | n]; [ simpl in Hrun; discriminate | ].
      rewrite run_succ, Hstep' in Hrun.
      destruct
        (IHn n (Nat.lt_succ_diag_r n) _ rs1 s1' (Nat.lt_succ_diag_r _)
           (related_step rs rs1 s' s1' Hrelated Hstep Herased1) Hrun)
        as [fuel Hfuel].
      exists (S fuel). rewrite region_run_succ, Hstep. exact Hfuel.
    + destruct n as [ | n]; [ simpl in Hrun; discriminate | ].
      rewrite run_succ, Hstep1' in Hrun.
      destruct n as [ | n]; [ simpl in Hrun; discriminate | ].
      rewrite run_succ, Hstep2' in Hrun.
      assert (Hn : n < S (S n)) by lia.
      destruct
        (IHn n Hn _ rs1 s2' (Nat.lt_succ_diag_r _)
           (related_step rs rs1 s' s2' Hrelated Hstep Herased1) Hrun)
        as [fuel Hfuel].
      exists (S fuel). rewrite region_run_succ, Hstep. exact Hfuel.
  - destruct Hsim as (vs0' & hp0 & Hstep' & Hvs).
    destruct n as [ | n]; [ simpl in Hrun; discriminate | ].
    rewrite run_succ, Hstep' in Hrun. injection Hrun as -> -> ->.
    exists 1. right. exists vs0, rs0. rewrite region_run_succ, Hstep.
    split; [ reflexivity | exact Hvs ].
  - exists 1. left. exists fault. rewrite region_run_succ, Hstep.
    reflexivity.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem erasure_exits : erasure_exits_statement.
Proof.
  intros e statics base fuel k vs rs Hfresh Herasable Hrun.
  exact
    (run_exits fuel _ _ k vs rs
       (related_initial e statics base Hfresh Herasable) Hrun).
Qed.

Theorem erasure_adds_no_exit : erasure_adds_no_exit_statement.
Proof.
  intros e statics base k vs' hp Hfresh Herasable [n Hrun].
  exact
    (run_adds_no_exit k vs' hp n _ _ _ (Nat.lt_succ_diag_r _)
       (related_initial e statics base Hfresh Herasable) Hrun).
Qed.
