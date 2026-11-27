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

(* The unit rebuilt with changes of calling convention binds its static
   sets whole wherever it runs to, as the plain rebuild does
   (FlambdaReaperTypesProofs, section [Rebuilt_sets]): the rebuild keeps
   the function slots of a set, and keeps or removes the bindings of a
   static set together. The proofs of that section, of
   FlambdaReaperCallingConventions.rebuild_expr_cc, whose bindings are
   rebuilt as the plain rebuild's, but code, and whose calls may be
   wrapped in a handler that applies a continuation. This file needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions.
From OxCamlFlambda2 Require FlambdaClosuresSpec FlambdaClosuresProofs
  FlambdaWellFormedProofs FlambdaReaperFlowProofs FlambdaReaperTypesProofs.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.

Section Rebuilt_sets_cc.

Variable F : facts.
Variable sol : solution.
Variable cc : node -> bool.
Variable code_deps : finite_map code_dep.
Variable b : bool.
Variable ids : list code_id.
Variable cmds : finite_map code_metadata.
Variable ff : positive.
Variable types : subkinds_rewriting.

Local Abbreviation rebuild :=
  (rebuild_expr_cc F sol cc code_deps b ids cmds ff types).
Local Abbreviation rebuild_h :=
  (rebuild_handler_cc F sol cc code_deps b ids cmds ff types).
Local Abbreviation rebuild_b :=
  (rebuild_binding_cc F sol cc code_deps b ids cmds ff types).

(* A binding rebuilt has the code ids and the symbols of the sets of the
   binding. *)
Lemma rebuild_binding_cc_shape (en : rebuild_env) (bd bd' : static_binding) :
  bd' ∈ rebuild_b en bd ->
  static_set_code_ids bd' = static_set_code_ids bd
  ∧ (∀ c : static_binding,
       binds_symbol_of_set bd' c = binds_symbol_of_set bd c)
  ∧ ∀ c : static_binding,
      binds_symbol_of_set c bd' = binds_symbol_of_set c bd.
Proof.
  destruct bd as [id c | id | s const]; cbn [rebuild_binding_cc].
  - destruct (is_code_id_used sol id); intros [<- | []];
      (split; [ reflexivity | split ]); intros c';
      destruct c' as [ | | s' [ ] ]; reflexivity.
  - intros [<- | []]. auto.
  - destruct (static_const_is_used sol s const); [ | intros [] ].
    intros [<- | []].
    split; [ destruct const; reflexivity | split ]; intros c';
      destruct const; destruct c' as [ | | s' [ ] ]; reflexivity.
Qed.

Lemma rebuild_bindings_cc_whole (en : rebuild_env)
    (bindings : list static_binding) :
  forallb (static_set_whole bindings) bindings = true ->
  forallb (static_set_whole (flat_map (rebuild_b en) bindings))
    (flat_map (rebuild_b en) bindings)
  = true.
Proof.
  intros H. apply forallb_forall. intros bd' Hin'.
  apply in_flat_map in Hin' as (bd & Hin & Hbd').
  pose proof (proj1 (forallb_forall _ _) H bd Hin) as Hw.
  destruct bd as [id c | id | s const]; cbn [rebuild_binding_cc] in Hbd'.
  - destruct (is_code_id_used sol id); destruct Hbd' as [<- | []];
      reflexivity.
  - destruct Hbd' as [<- | []]. reflexivity.
  - destruct (static_const_is_used sol s const) eqn:Hused;
      [ | destruct Hbd' ].
    destruct Hbd' as [<- | []].
    destruct const as [code vss | slot syms decls vss | tag fields | contents
                      | bytes | elements];
      try reflexivity.
    cbn [rebuild_static_const static_set_whole] in Hw |- *.
    apply andb_true_iff in Hw as [Hw Hall]. rewrite Hw. cbn [andb].
    apply forallb_forall. intros [slot' s'] Hin_s.
    pose proof (proj1 (forallb_forall _ _) Hall (slot', s') Hin_s) as Hex.
    cbn beta iota in Hex.
    apply existsb_exists in Hex as (bd2 & Hbd2 & Hbinds).
    apply FlambdaClosuresProofs.binds_symbol_to_sound in Hbinds.
    subst bd2.
    apply existsb_exists.
    eexists. split.
    + apply in_flat_map.
      exists (Bind_symbol s' (Static_const_closure_of_set slot' syms decls
                                vss)).
      split; [ exact Hbd2 | ].
      cbn [rebuild_binding_cc]. cbn [static_const_is_used] in Hused |- *.
      rewrite Hused. left. reflexivity.
    + cbn [rebuild_static_const binds_symbol_to]. rewrite Pos.eqb_refl.
      destruct (static_const_eq_dec _ _) as [_ | Hne];
        [ reflexivity | exfalso; apply Hne; reflexivity ].
Qed.

Lemma rebuild_cc_let_static_unfold (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  rebuild en (Let_static bindings body)
  = Let_static (flat_map (rebuild_b en) bindings)
      (rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma rebuild_cc_sets_well_formed :
  (∀ (e : expr) (en : rebuild_env),
     sets_well_formed e = true -> sets_well_formed (rebuild en e) = true)
  ∧ ∀ (h : continuation_handler) (en : rebuild_env) (dc : bool)
      (mask : list bool),
      handler_sets_well_formed h = true ->
      handler_sets_well_formed (rebuild_h en dc mask h) = true.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH en H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H as [Hd Hb].
    cbn [rebuild_expr_cc].
    destruct (is_pending_projection en x d).
    + cbn [sets_well_formed]. rewrite Hd. exact (IH _ Hb).
    + destruct d; cbn [named_sets_well_formed] in Hd;
        repeat match goal with
               | |- sets_well_formed (if ?c then _ else _) = true =>
                 destruct c
               end;
        try reflexivity; try exact (IH _ Hb);
        cbn [sets_well_formed]; rewrite (IH _ Hb), andb_true_r;
        [ .. | exact Hd ];
        unfold rebuild_named;
        repeat match goal with
               | |- named_sets_well_formed (match ?c with _ => _ end) = true
                 => destruct c
               | |- named_sets_well_formed (if ?c then _ else _) = true =>
                 destruct c
               end;
        reflexivity.
  - intros bindings body Hcodes IH en H.
    rewrite TP.sets_well_formed_let_static in H.
    apply andb_true_iff in H as [H Hb].
    apply andb_true_iff in H as [H Hbs].
    apply andb_true_iff in H as [Hwhole Hdisj].
    rewrite rebuild_cc_let_static_unfold, TP.sets_well_formed_let_static.
    rewrite (rebuild_bindings_cc_whole en bindings Hwhole).
    rewrite (TP.code_disjoint_flat_map (rebuild_b en)); [ | | | exact Hdisj ].
    + rewrite (IH _ Hb), !andb_true_r. cbn [andb].
      apply forallb_forall. intros bd' Hin'.
      apply in_flat_map in Hin' as (bd & Hin & Hbd').
      pose proof (proj1 (forallb_forall _ _) Hbs bd Hin) as Hw.
      destruct bd as [id c | id | s const]; cbn [rebuild_binding_cc] in Hbd'.
      * destruct (is_code_id_used sol id); destruct Hbd' as [<- | []];
          [ | reflexivity ].
        pose proof (Hcodes id c Hin) as Hc.
        destruct c as [params mc ret exn md cbody].
        cbn [binding_sets_well_formed code_sets_well_formed code_body
             rebuild_code_cc] in Hw, Hc |- *.
        destruct (code_calling_convention_change _ _ _ _ _ _);
          cbn [code_sets_well_formed];
          (destruct (existsb _ _); [ reflexivity | exact (Hc _ Hw) ]).
      * destruct Hbd' as [<- | []]. reflexivity.
      * destruct (static_const_is_used sol s const); [ | destruct Hbd' ].
        destruct Hbd' as [<- | []]. reflexivity.
    + intros bd. destruct bd as [id c | id | s const];
        cbn [rebuild_binding_cc].
      * destruct (is_code_id_used sol id); right; eexists; reflexivity.
      * right. eexists. reflexivity.
      * destruct (static_const_is_used sol s const);
          [ right; eexists; reflexivity | left; reflexivity ].
    + exact (rebuild_binding_cc_shape en).
  - intros k h body IHh IH en H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H as [Hh Hb].
    cbn [rebuild_expr_cc].
    destruct (among k _); [ | exact (IH _ Hb) ].
    cbn [sets_well_formed]. rewrite (IHh _ _ _ Hh). exact (IH _ Hb).
  - intros handlers body IHh IH en H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H as [Hh Hb].
    cbn [rebuild_expr_cc sets_well_formed]. rewrite (IH _ Hb), andb_true_r.
    apply forallb_forall. intros [k h'] Hin.
    apply in_map_iff in Hin as ([k0 h0] & Heq & Hin). injection Heq as <- <-.
    apply IHh with (k := k0); [ exact Hin | ].
    exact (proj1 (forallb_forall _ _) Hh (k0, h0) Hin).
  - intros ap en _. cbn [rebuild_expr_cc]. unfold rebuild_apply_cc.
    destruct (apply_calling_convention_change _ _ _ _ _ _ ap).
    + unfold rebuild_apply.
      destruct (_ || _); [ reflexivity | ]. unfold make_apply_wrapper.
      destruct (lookup _ _); [ destruct (forallb _ _) | ]; reflexivity.
    + destruct (_ || _); [ reflexivity | ]. unfold make_apply_wrapper_cc.
      destruct (forallb _ _); reflexivity.
  - intros ac en _. cbn [rebuild_expr_cc]. unfold rebuild_apply_cont.
    destruct (rewrite_apply_cont _ _ _); reflexivity.
  - intros scrutinee arms en _. cbn [rebuild_expr_cc]. unfold rebuild_switch.
    destruct (Keyed.is_empty _); reflexivity.
  - intros en _. reflexivity.
  - intros params metadata handler IH en dc mask H.
    cbn [rebuild_handler_cc handler_sets_well_formed] in H |- *.
    destruct (dc && _); [ reflexivity | exact (IH _ H) ].
Qed.

End Rebuilt_sets_cc.

(* The unit rebuilt with changes of calling convention, of a well-formed
   unit, binds its static sets whole wherever it runs to. *)
Lemma cc_rebuilt_sets_bound (types : subkinds_rewriting) (F : facts)
    (sol : solution) (cc : node -> bool) (code_deps : finite_map code_dep)
    (b : bool) (u : unit_interface) (e : expr) (m : nat) (t : state) :
  well_formed_unit u e = true ->
  run m
    (State (rebuild_unit_cc_with types F sol cc code_deps b e) [] []
       empty_static_env empty_heap no_reads)
  = Step t ->
  FlambdaClosuresSpec.static_sets_bound (state_statics t).
Proof.
  intros Hwf Hm.
  assert (Hsets :
            sets_well_formed (rebuild_unit_cc_with types F sol cc code_deps b
                                e)
            = true).
  { unfold rebuild_unit_cc_with.
    apply (proj1 (rebuild_cc_sets_well_formed F sol cc code_deps b _ _ _
                    types)).
    exact (FlambdaWellFormedProofs.well_formed_unit_sets_well_formed u e
             Hwf). }
  exact (proj2 (proj2 (proj2
           (FlambdaClosuresProofs.sets_inv_run m _ t
              (FlambdaClosuresProofs.sets_inv_initial _ Hsets) Hm)))).
Qed.
