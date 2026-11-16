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

(* Proofs of the statements of FlambdaRangeSpec.v. Nothing here needs
   review.

   The primitives that compute on values alone are not looked into:
   what one does to numbers in range is the law
   FlambdaPrimitiveProofs.eval_prim_in_range. Those that read or change
   the heap are looked into here, case by case ([exec_prim_in_range],
   by the cases of FlambdaHeapSpec.exec_bound_prim). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaNominalSpec
  FlambdaEquivarianceTactic FlambdaRangeSpec.
From OxCamlFlambda2 Require FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* Evaluation                                                               *)
(* ======================================================================== *)

Definition slot_values_in_range (value_slots : list (value_slot * value))
    : bool :=
  forallb (fun '(_, v) => value_in_range v) value_slots.

Section Evaluation.

Variable vars : env.
Hypothesis Hvars : env_in_range vars.

Lemma eval_simple_in_range (s : simple) (v : value) :
  simple_in_range s = true ->
  eval_simple vars s = Some v ->
  value_in_range v = true.
Proof.
  destruct s as [x | sym | c]; simpl.
  - intros _ Hlookup. exact (Hvars x v (lookup_in _ _ _ Hlookup)).
  - intros _ [= <-]. reflexivity.
  - intros Hc [= <-]. exact Hc.
Qed.

Lemma eval_simples_in_range (args : list simple) :
  ∀ vs : list value,
    forallb simple_in_range args = true ->
    eval_simples vars args = Some vs ->
    forallb value_in_range vs = true.
Proof.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_in_range s v Hs Heval), (IH vs' Hargs);
      reflexivity.
Qed.

Lemma eval_value_slots_in_range (value_slots : list (value_slot * simple)) :
  ∀ evaluated : list (value_slot * value),
    value_slots_in_range value_slots = true ->
    eval_value_slots vars value_slots = Some evaluated ->
    slot_values_in_range evaluated = true.
Proof.
  unfold value_slots_in_range, slot_values_in_range.
  induction value_slots as [ | [slot s] value_slots IH]; intros evaluated;
    simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hslots. apply andb_true_iff in Hslots as [Hs Hslots].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_in_range s v Hs Heval), (IH rest Hslots);
      reflexivity.
Qed.

End Evaluation.

Lemma lookup_slot_in_range (slot : value_slot)
    (value_slots : list (value_slot * value)) (v : value) :
  slot_values_in_range value_slots = true ->
  lookup slot value_slots = Some v ->
  value_in_range v = true.
Proof.
  intros Hslots Hlookup.
  exact
    (proj1 (forallb_forall _ _) Hslots (slot, v) (lookup_in _ _ _ Hlookup)).
Qed.

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

Section Statics.

Variable statics : static_env.
Hypothesis Hstatics : statics_in_range statics.

Lemma lookup_symbol_in_range (s : symbol)
    (allocated : statically_allocated_value) :
  lookup s (static_values statics) = Some allocated ->
  allocated_in_range allocated = true.
Proof.
  intros Hlookup. exact (proj2 Hstatics s allocated (lookup_in _ _ _ Hlookup)).
Qed.

(* The value slots of a closure in range are, whether the closure was
   made by a [Let] or is what a symbol is bound to. *)
Lemma closure_in_set_in_range (v : value) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  value_in_range v = true ->
  closure_in_set statics v = Some (slot, function_decls, value_slots) ->
  slot_values_in_range value_slots = true.
Proof.
  intros Hv. destruct v; simpl; try discriminate.
  - intros [= _ _ <-]. exact Hv.
  - match goal with
    | |- context [lookup ?at_slot ?decls] => destruct (lookup at_slot decls)
    end;
      [ | discriminate ].
    simpl. intros [= _ _ <-]. exact Hv.
  - match goal with
    | |- context [lookup ?s ?values] =>
      destruct (lookup s values) as [allocated | ] eqn:Hlookup;
        [ | discriminate ];
        pose proof (lookup_symbol_in_range s allocated Hlookup)
          as Hallocated
    end.
    destruct allocated; try discriminate.
    + intros [= _ _ <-]. exact Hallocated.
    + match goal with
      | |- context [lookup ?at_slot ?decls] =>
        destruct (lookup at_slot decls)
      end;
        [ | discriminate ].
      simpl. intros [= _ _ <-]. exact Hallocated.
Qed.

(* The closure of the same set at another slot has the same value slots,
   and a symbol is in range. *)
Lemma project_function_slot_in_range (move_from move_to : function_slot)
    (v w : value) :
  value_in_range v = true ->
  project_function_slot statics move_from move_to v = Some w ->
  value_in_range w = true.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v; try discriminate.
  - destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hv.
  - match goal with
    | |- context [Pos.eqb ?at_slot move_from] =>
      destruct (Pos.eqb at_slot move_from); [ | discriminate ]
    end.
    repeat
      match goal with
      | |- context [lookup ?at_slot ?decls] =>
        destruct (lookup at_slot decls); [ | discriminate ]
      end.
    simpl. intros [= <-]. exact Hv.
  - match goal with
    | |- context [lookup ?s ?values] =>
      destruct (lookup s values) as [allocated | ]; [ | discriminate ]
    end.
    destruct allocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. reflexivity.
    + match goal with
      | |- context [Pos.eqb ?at_slot move_from] =>
        destruct (Pos.eqb at_slot move_from); [ | discriminate ]
      end.
      repeat
        match goal with
        | |- context [lookup ?key ?bindings] =>
          destruct (lookup key bindings); [ | discriminate ]
        end.
      simpl. intros [= <-]. reflexivity.
Qed.

Variable vars : env.
Hypothesis Hvars : env_in_range vars.

Lemma eval_named_in_range (d : named) (v : value) :
  named_in_range d = true ->
  eval_named statics vars d = Some v ->
  value_in_range v = true.
Proof.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure];
    simpl; intros Hd.
  - exact (eval_simple_in_range vars Hvars s v Hd).
  - apply andb_true_iff in Hd as [Hp Hargs].
    destruct (eval_simples vars args) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl.
    pose proof (eval_simples_in_range vars Hvars args vs Hargs Heval) as Hvs.
    apply (eval_prim_in_range statics p vs v Hp).
    + intros sym allocated _. apply lookup_symbol_in_range.
    + apply forallb_forall. exact Hvs.
  - destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_in_range vars Hvars slots evaluated Hd Heval).
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_in_range vars Hvars slots evaluated Hd Heval).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl.
    destruct (closure_of statics w) as [[called value_slots] | ] eqn:Hof;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics w called value_slots) Hof)
      as (at_slot & function_decls & Hin_set & _).
    exact
      (lookup_slot_in_range slot value_slots v
         (closure_in_set_in_range w at_slot function_decls value_slots
            (eval_simple_in_range vars Hvars closure w Hd Hclosure) Hin_set)
         Hlookup).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hproject.
    destruct (proj1 (project_value_slot_defined statics pfrom slot w v)
                Hproject)
      as (function_decls & value_slots & Hin_set & Hlookup).
    exact
      (lookup_slot_in_range slot value_slots v
         (closure_in_set_in_range w pfrom function_decls value_slots
            (eval_simple_in_range vars Hvars closure w Hd Hclosure) Hin_set)
         Hlookup).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl.
    exact
      (project_function_slot_in_range move_from move_to w v
         (eval_simple_in_range vars Hvars closure w Hd Hclosure)).
Qed.

End Statics.

(* ======================================================================== *)
(* A primitive on a heap in range                                           *)
(* ======================================================================== *)

Lemma heap_in_range_write (l : location) (c : cell) (h : heap) :
  heap_in_range h -> cell_in_range c = true -> heap_in_range ((l, c) :: h).
Proof.
  intros Hheap Hc l0 c0 [[= <- <-] | Hin];
    [ exact Hc | exact (Hheap l0 c0 Hin) ].
Qed.

(* The cases are those of FlambdaHeapProofs.exec_prim_preserves. *)
Theorem exec_prim_in_range : exec_prim_in_range_statement.
Proof.
  intros st h h1 p vs result Hp Hsymbols Hvs Hheap Hexec.
  apply FlambdaHeapProofs.exec_prim_bound in Hexec as [_ Hexec].
  revert Hexec.
  assert (Hpure :
            (let* v := eval_prim st p vs in Some (v, h)) = Some (result, h1) ->
            value_in_range result = true ∧ heap_in_range h1).
  { destruct (eval_prim st p vs) as [v | ] eqn:Heval; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ | exact Hheap ].
    exact (eval_prim_in_range st p vs v Hp Hsymbols Hvs Heval). }
  assert (Hcell :
            ∀ (l : location) (c : cell),
              lookup l h = Some c -> cell_in_range c = true).
  { intros l c Hl. exact (Hheap l c (lookup_in _ _ _ Hl)). }
  destruct p as [p | p | p | p | rp]; try exact Hpure.
  - destruct vs as [ | v [ | v2 vs]]; try exact Hpure.
    2: { destruct v; simpl; discriminate. }
    rewrite FlambdaHeapProofs.exec_bound_prim_unary.
    destruct (FlambdaHeapProofs.location_of v) as [l | ] eqn:Hlocation.
    2: { destruct v; try discriminate; exact Hpure. }
    destruct v; try discriminate. injection Hlocation as ->.
    destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
    destruct (exec_unary_on_cell p l c) as [res | ] eqn:Hres; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ | exact Hheap ].
    pose proof (Hcell l c Hc) as Hrange.
    destruct c as [cell_mut tag fields | akind elements]; simpl in Hrange.
    + apply andb_true_iff in Hrange as [Htag Hfields].
      destruct p; simpl in Hres; try discriminate.
      * destruct (load_allowed _ cell_mut); [ | discriminate ].
        apply FlambdaHeapProofs.load_field_in in Hres.
        exact (proj1 (forallb_forall _ _) Hfields res Hres).
      * injection Hres as <-. apply naked_bool_in_range.
      * injection Hres as <-. apply naked_bool_in_range.
      * injection Hres as <-. simpl.
        exact (tag_in_range_ocaml_int tag Htag).
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. reflexivity.
    + destruct p as [ | [ | ] | | | length_kind | | okind | | | | | | | | ];
        simpl in Hres; try discriminate.
      * injection Hres as <-. apply naked_bool_in_range.
      * injection Hres as <-. apply naked_bool_in_range.
      * unfold array_length_of in Hres.
        destruct (in_range _ _) eqn:Hlength; [ | discriminate ].
        injection Hres as <-. exact Hlength.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. reflexivity.
  - rewrite FlambdaHeapProofs.exec_bound_prim_binary.
    destruct (FlambdaHeapProofs.phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location st v1 v2); simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ apply naked_bool_in_range | exact Hheap ].
    + destruct (FlambdaHeapProofs.binary_on_cell p vs) as [[l v2] | ];
        [ | exact Hpure ].
      destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (exec_binary_on_cell p l c v2) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ | exact Hheap ].
      pose proof (Hcell l c Hc) as Hrange.
      destruct c as [cell_mut tag fields | akind elements];
        [ destruct p; simpl in Hres; discriminate | ].
      destruct p as [array_kind load_kind mut | | | | | ]; simpl in Hres;
        try discriminate.
      destruct (load_allowed mut Cell_mutable); [ | discriminate ].
      apply array_loaded_element in Hres as [Hin _].
      exact (proj1 (forallb_forall _ _) Hrange res Hin).
  - change (exec_bound_prim st h (Heap p) vs)
      with (exec_heap_primitive st h p vs).
    destruct p as [mut tag | kind init field | kind | akind set_kind].
    + simpl. destruct (forallb _ vs) eqn:Hkinds; [ | discriminate ].
      intros [= <- <-]. split; [ reflexivity | ].
      apply heap_in_range_write; [ exact Hheap | ].
      simpl. simpl in Hp. rewrite Hp. simpl.
      apply forallb_forall. exact Hvs.
    + destruct vs as [ | v [ | new [ | v3 vs]]]; try (simpl; discriminate).
      rewrite FlambdaHeapProofs.exec_block_set.
      destruct (FlambdaHeapProofs.location_of v) as [l | ];
        [ | discriminate ].
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
        eqn:Hc;
        try discriminate.
      destruct (store_field kind field new tag fields) as [fields1 | ]
        eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (FlambdaHeapProofs.store_field_in _ _ _ _ _ _ Hstore)
        as [_ Hin].
      pose proof (Hcell l _ Hc) as Hrange. simpl in Hrange.
      apply andb_true_iff in Hrange as [Htag Hfields].
      split; [ reflexivity | ].
      apply heap_in_range_write; [ exact Hheap | ].
      simpl. rewrite Htag. simpl. apply forallb_forall. intros v0 Hv0.
      destruct (Hin v0 Hv0) as [-> | Hold].
      * apply Hvs. right. left. reflexivity.
      * exact (proj1 (forallb_forall _ _) Hfields v0 Hold).
    + simpl. destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
        simpl; [ | discriminate ].
      intros [= <- <-].
      apply array_made_of_arguments in Hmade as [-> Hok].
      split; [ reflexivity | ].
      apply heap_in_range_write; [ exact Hheap | ].
      unfold cell_in_range. apply forallb_forall. exact Hvs.
    + destruct vs as [ | v [ | index [ | new [ | v4 vs]]]];
        try (simpl; discriminate).
      rewrite FlambdaHeapProofs.exec_array_set.
      destruct (FlambdaHeapProofs.location_of v) as [l | ];
        [ | discriminate ].
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hc;
        try discriminate.
      destruct (array_stored set_kind kind elements index new)
        as [stored | ] eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (array_stored_elements _ _ _ _ _ _ Hstore) as (_ & _ & Hin).
      pose proof (Hcell l _ Hc) as Hrange. simpl in Hrange.
      split; [ reflexivity | ].
      apply heap_in_range_write; [ exact Hheap | ].
      simpl. apply forallb_forall. intros v0 Hv0.
      destruct (Hin v0 Hv0) as [-> | Hold].
      * apply Hvs. right. right. left. reflexivity.
      * exact (proj1 (forallb_forall _ _) Hrange v0 Hold).
Qed.

(* What a named expression does on a heap in range: where it has a
   value, the value is in range and so is the heap after. *)
Lemma exec_named_in_range (statics : static_env)
    (Hstatics : statics_in_range statics) (vars : env)
    (Hvars : env_in_range vars) (hp hp' : heap) (rd rd' : read_record)
    (d : named) (v : value) :
  named_in_range d = true ->
  heap_in_range hp ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  value_in_range v = true ∧ heap_in_range hp'.
Proof.
  intros Hd Hheap Hexec.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named statics vars _) as [w | ] eqn:Heval;
           [ | discriminate ];
         injection Hexec as <- <- <-;
         split; [ | exact Hheap ];
         exact (eval_named_in_range statics Hstatics vars Hvars _ w Hd Heval)).
  apply exec_named_prim_some in Hexec as (vs & Heval & Hexec & _).
  simpl in Hd. apply andb_true_iff in Hd as [Hp Hargs].
  pose proof (eval_simples_in_range vars Hvars args vs Hargs Heval) as Hvs.
  apply (exec_prim_in_range statics hp hp' p vs v Hp); try assumption.
  - intros sym allocated _.
    exact (lookup_symbol_in_range statics Hstatics sym allocated).
  - apply forallb_forall. exact Hvs.
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Lemma eval_static_const_in_range (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_in_range vars ->
  static_const_in_range const = true ->
  eval_static_const vars const = Some allocated ->
  allocated_in_range allocated = true.
Proof.
  intros Hvars.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl; intros Hconst.
  - destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_in_range vars Hvars slots evaluated Hconst Heval).
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_in_range vars Hvars slots evaluated Hconst Heval).
  - apply andb_true_iff in Hconst as [Htag Hfields].
    destruct (eval_simples vars fields) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Htag.
    exact (eval_simples_in_range vars Hvars fields vs Hfields Heval).
  - destruct (eval_simple vars contents) as [v | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_in_range vars Hvars contents v Hconst Heval).
  - intros [= <-]. exact Hconst.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_in_range vars Hvars elements vs Hconst Heval).
Qed.

Lemma eval_static_bindings_in_range (vars : env)
    (bindings : list static_binding) :
  env_in_range vars ->
  ∀ statics statics' : static_env,
    forallb binding_numbers_in_range bindings = true ->
    statics_in_range statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    statics_in_range statics'.
Proof.
  intros Hvars.
  induction bindings as [ | binding bindings IH]; intros statics statics';
    simpl.
  - intros _ Hstatics [= <-]. exact Hstatics.
  - intros Hbindings. apply andb_true_iff in Hbindings as [Hbinding Hrest].
    intros [Hcode Hvalues].
    destruct binding as [id c | id | s const]; simpl in Hbinding.
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      apply (IH _ _ Hrest). split; [ | exact Hvalues ].
      intros id' c' [[= <- <-] | Hin]; [ exact Hbinding | ].
      exact (Hcode id' c' Hin).
    + apply (IH _ _ Hrest). split; assumption.
    + destruct (lookup s (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Heval;
        [ | discriminate ].
      simpl. apply (IH _ _ Hrest). split; [ exact Hcode | ].
      intros s' allocated' [[= <- <-] | Hin].
      * exact
          (eval_static_const_in_range vars const allocated Hvars Hbinding
             Heval).
      * exact (Hvalues s' allocated' Hin).
Qed.

(* ======================================================================== *)
(* Continuations and calls                                                  *)
(* ======================================================================== *)

Lemma resolve_in_range (k : continuation) (conts : cont_env) :
  conts_in_range conts -> cont_in_range (resolve k conts).
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hlookup.
  - exact (Hconts k c (lookup_in _ _ _ Hlookup)).
  - constructor.
Qed.

Lemma handlers_in_range
    (handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_numbers_in_range h) handlers = true ->
  ∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers -> handler_numbers_in_range h = true.
Proof.
  intros Hhandlers k h Hin.
  exact (proj1 (forallb_forall _ _) Hhandlers (k, h) Hin).
Qed.

Lemma rec_closures_in_range
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  (∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers -> handler_numbers_in_range h = true) ->
  env_in_range vars ->
  conts_in_range conts ->
  conts_in_range (rec_closures handlers vars conts).
Proof.
  intros Hhandlers Hvars Hconts k c Hin. unfold rec_closures in Hin.
  apply in_map_iff in Hin as ([k' h] & [= <- <-] & Hin).
  constructor; [ exact (Hhandlers k' h Hin) | assumption .. ].
Qed.

Lemma conts_in_range_app (conts conts' : cont_env) :
  conts_in_range conts ->
  conts_in_range conts' ->
  conts_in_range (conts ++ conts').
Proof.
  intros Hconts Hconts' k c Hin.
  apply in_app_or in Hin as [Hin | Hin]; eauto.
Qed.

Lemma bind_params_in_range (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    forallb value_in_range vs = true ->
    env_in_range vars ->
    bind_params params vs vars = Some vars' ->
    env_in_range vars'.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars'; simpl;
    try discriminate.
  - intros _ Hvars [= <-]. exact Hvars.
  - intros Hvs. apply andb_true_iff in Hvs as [Hv Hvs]. intros Hvars.
    apply (IH vs _ _ Hvs).
    intros y w [[= <- <-] | Hin]; [ exact Hv | exact (Hvars y w Hin) ].
Qed.

(* What applying a continuation comes to: a state in range, or a way out
   with values in range. *)
Definition result_in_range (result : step_result) : Prop :=
  match result with
  | Step s => state_in_range s
  | Exit _ vs _ _ => forallb value_in_range vs = true
  | Undefined => True
  end.

(* A value in range, as a property: that of a partial application is
   that of the value it calls and of its arguments. *)
Lemma value_in_range_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_in_range (Value_partial_application callee args args_arity) = true
  <-> value_in_range callee = true
      ∧ Forall (fun v => value_in_range v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* The fields of a tuple in range are in range, where what the symbols
   are bound to is. *)
Lemma tuple_fields_in_range (statics : static_env) :
  statics_in_range statics ->
  P_fields (fun v => value_in_range v = true) statics.
Proof.
  intros Hstatics size tuple fields Htuple Hfields.
  apply (tuple_fields_of_blocks _ statics size tuple fields);
    [ | exact Hfields ].
  intros tag block Hblock. apply forallb_Forall.
  destruct (block_of_cases _ _ _ _ Hblock) as [-> | (s & -> & Hs)].
  - simpl in Htuple. apply andb_true_iff in Htuple as [_ Htuple].
    exact Htuple.
  - pose proof (proj2 Hstatics s _ (lookup_in _ _ _ Hs)) as Hallocated.
    simpl in Hallocated. apply andb_true_iff in Hallocated as [_ Hallocated].
    exact Hallocated.
Qed.

Lemma call_code_in_range (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  value_in_range closure = true ->
  forallb value_in_range vs = true ->
  cont_in_range return_to ->
  cont_in_range exn_to ->
  statics_in_range statics ->
  heap_in_range hp ->
  lookup called (static_code statics) = Some c ->
  result_in_range
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd).
Proof.
  intros Hclosure Hvs Hreturn Hexn Hstatics Hheap Hcode. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ] eqn:Hbind;
    [ | exact I ].
  pose proof (proj1 Hstatics called c (lookup_in _ _ _ Hcode)) as Hc.
  repeat split; simpl.
  - destruct c. exact Hc.
  - apply (bind_params_in_range _ _ _ _ Hvs) in Hbind; [ exact Hbind | ].
    intros y w [[= <- <-] | []]. exact Hclosure.
  - intros k c' [[= <- <-] | [[= <- <-] | []]]; assumption.
  - exact (proj1 Hstatics).
  - exact (proj2 Hstatics).
  - exact Hheap.
Qed.

Lemma run_call_in_range (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  shape_calls statics kind shape ->
  shape_values_all (fun v => value_in_range v = true) shape ->
  cont_in_range return_to ->
  cont_in_range exn_to ->
  statics_in_range statics ->
  heap_in_range hp ->
  (∀ partial : value,
    value_in_range partial = true ->
    result_in_range (return_partial partial)) ->
  result_in_range
    (run_call return_partial shape return_arity return_to exn_to statics hp rd).
Proof.
  intros Hcalls Hvalues Hreturn Hexn Hstatics Hheap Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls, Hvalues |- *.
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    destruct Hvalues as [Hclosure Hvs]. apply forallb_Forall in Hvs.
    exact (call_code_in_range closure called c vs _ _ _ _ statics _ rd_ Hclosure
             Hvs Hreturn Hexn Hstatics Hheap Hcode).
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    destruct Hvalues as (Hclosure & Hvs & Hremaining).
    apply forallb_Forall in Hvs, Hremaining.
    apply (call_code_in_range closure called c vs _ _ _ _ statics _ rd_ Hclosure
             Hvs);
      [ constructor; assumption | exact Hexn | exact Hstatics | exact Hheap
      | exact Hcode ].
  - exact (Hpartial partial Hvalues).
  - exact I.
Qed.

Lemma apply_cont_value_in_range (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_in_range c ->
  forallb value_in_range vs = true ->
  statics_in_range statics ->
  heap_in_range hp ->
  result_in_range (apply_cont_value c vs statics hp rd).
Proof.
  intros Hc Hvs Hstatics Hheap. revert vs Hvs.
  induction Hc
    as [k | c extra Hc IH Hextra
       | [params metadata handler] vars conts recursive Hh Hvars Hconts _
           Hrecursive
       | remaining remaining_arity return_arity return_to exn_to Hremaining
           Hreturn IHreturn Hexn _];
    intros vs Hvs.
  - exact Hvs.
  - simpl. apply IH. rewrite forallb_app, Hvs, Hextra. reflexivity.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    repeat split; simpl.
    + exact Hh.
    + exact
        (bind_params_in_range params vs vars handler_vars Hvs Hvars Hbind).
    + apply conts_in_range_app; [ | exact Hconts ].
      exact
        (rec_closures_in_range recursive vars conts Hrecursive Hvars Hconts).
    + exact (proj1 Hstatics).
    + exact (proj2 Hstatics).
    + exact Hheap.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    simpl in Hvs. rewrite andb_true_r in Hvs. unfold call_function.
    apply (run_call_in_range _ Indirect_unknown_arity); try assumption.
    + apply shape_of_call_calls.
    + apply (shape_of_call_values_all _ value_in_range_partial);
        [ exact (tuple_fields_in_range statics Hstatics) | exact Hvs
        | apply forallb_Forall; exact Hremaining ].
    + intros partial Hpartial. apply IHreturn. simpl.
      rewrite Hpartial. reflexivity.
Qed.

Lemma with_extra_args_in_range (c : cont_value) (extra : list value) :
  cont_in_range c ->
  forallb value_in_range extra = true ->
  cont_in_range (with_extra_args c extra).
Proof.
  intros Hc Hextra. destruct extra as [ | v extra]; simpl;
    [ exact Hc | constructor; assumption ].
Qed.

Lemma call_function_in_range (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  value_in_range callee = true ->
  forallb value_in_range vs = true ->
  cont_in_range return_to ->
  cont_in_range exn_to ->
  statics_in_range statics ->
  heap_in_range hp ->
  result_in_range
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd).
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics Hheap. unfold call_function.
  apply (run_call_in_range _ kind); try assumption.
  - apply shape_of_call_calls.
  - apply (shape_of_call_values_all _ value_in_range_partial);
      [ exact (tuple_fields_in_range statics Hstatics) | exact Hcallee
      | apply forallb_Forall; exact Hvs ].
  - intros partial Hpartial.
    apply apply_cont_value_in_range;
      [ exact Hreturn | | exact Hstatics | exact Hheap ].
    simpl. rewrite Hpartial. reflexivity.
Qed.

(* ======================================================================== *)
(* Steps and runs                                                           *)
(* ======================================================================== *)

Lemma step_result_in_range (s : state) :
  state_in_range s -> result_in_range (step s).
Proof.
  intros (Hexpr & Hvars & Hconts & Hstatics & Hheap). rewrite (state_eta s).
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *. set (hp := state_heap s) in *.
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step; simpl in Hexpr.
  - apply andb_true_iff in Hexpr as [Hd Hbody].
    destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ] eqn:Heval;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct
      (exec_named_in_range statics Hstatics vars Hvars _ _ _ _ d v Hd Hheap
         Heval)
      as [Hv Hheap'].
    repeat split; simpl; try assumption; try apply Hstatics.
    intros y w [[= <- <-] | Hin]; [ exact Hv | exact (Hvars y w Hin) ].
  - apply andb_true_iff in Hexpr as [Hbindings Hbody].
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval;
      [ | exact I ].
    pose proof
      (eval_static_bindings_in_range vars bindings Hvars statics extended
         Hbindings Hstatics Heval)
      as Hextended.
    repeat split; simpl; try assumption; apply Hextended.
  - apply andb_true_iff in Hexpr as [Hh Hbody].
    repeat split; simpl; try assumption; try apply Hstatics.
    intros k' c [[= <- <-] | Hin]; [ | exact (Hconts k' c Hin) ].
    constructor; try assumption. intros k' h' [].
  - apply andb_true_iff in Hexpr as [Hhandlers Hbody].
    destruct (handlers_bind_statics handlers); [ exact I | ].
    repeat split; simpl; try assumption; try apply Hstatics.
    apply conts_in_range_app; [ | exact Hconts ].
    exact
      (rec_closures_in_range handlers vars conts
         (handlers_in_range handlers Hhandlers) Hvars Hconts).
  - apply andb_true_iff in Hexpr as [Hexpr Hextra].
    apply andb_true_iff in Hexpr as [Hcallee Hargs].
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hc;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextras; [ | exact I ].
    apply call_function_in_range; try assumption.
    + exact (eval_simple_in_range vars Hvars _ callee Hcallee Hc).
    + exact (eval_simples_in_range vars Hvars _ vs Hargs Hvs).
    + apply resolve_in_range, Hconts.
    + apply with_extra_args_in_range; [ apply resolve_in_range, Hconts | ].
      exact (eval_simples_in_range vars Hvars _ extra Hextra Hextras).
  - destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    apply apply_cont_value_in_range; try assumption.
    + apply resolve_in_range, Hconts.
    + exact (eval_simples_in_range vars Hvars _ vs Hexpr Hvs).
  - apply andb_true_iff in Hexpr as [_ Harms].
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | exact I ].
    pose proof
      (proj1 (forallb_forall _ _) Harms (i, ac) (find_arm_in i arms ac Hfind))
      as Harm.
    apply andb_true_iff in Harm as [_ Hac].
    repeat split; simpl; try assumption; apply Hstatics.
  - exact I.
Qed.

Theorem step_in_range : step_in_range_statement.
Proof.
  intros s s' Hs Hstep. pose proof (step_result_in_range s Hs) as Hresult.
  rewrite Hstep in Hresult. exact Hresult.
Qed.

Lemma run_result_in_range (n : nat) :
  ∀ s : state, state_in_range s -> result_in_range (run n s).
Proof.
  induction n as [ | n IH]; intros s Hs; simpl; [ exact Hs | ].
  pose proof (step_result_in_range s Hs) as Hresult.
  destruct (step s) as [next | k vs | ]; [ exact (IH next Hresult) | .. ];
    exact Hresult.
Qed.

Lemma initial_state_in_range (e : expr) :
  numbers_in_range e = true ->
  state_in_range (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros He. repeat split; simpl; try exact He; intros ? ? [].
Qed.

Theorem run_in_range : run_in_range_statement.
Proof.
  intros e n s He Hrun.
  pose proof (run_result_in_range n _ (initial_state_in_range e He))
    as Hresult.
  rewrite Hrun in Hresult. exact Hresult.
Qed.

Theorem run_exit_in_range : run_exit_in_range_statement.
Proof.
  intros e n k vs hp rd He Hrun.
  pose proof (run_result_in_range n _ (initial_state_in_range e He))
    as Hresult.
  rewrite Hrun in Hresult. apply forallb_forall. exact Hresult.
Qed.

Theorem well_formed_in_range : well_formed_in_range_statement.
Proof.
  intros u e n s Hwf.
  exact (run_in_range e n s (well_formed_unit_numbers_in_range u e Hwf)).
Qed.

(* ======================================================================== *)
(* What a state in range holds                                              *)
(* ======================================================================== *)

Theorem blocks_in_range : blocks_in_range_statement.
Proof.
  intros s x v tag fields (_ & Hvars & _ & Hstatics & _) Hlookup Hblock.
  destruct
    (block_of_in_range (state_statics s) v tag fields) as [Htag Hfields];
    [ | exact (Hvars x v (lookup_in _ _ _ Hlookup)) | exact Hblock | ].
  - intros sym allocated _.
    exact (lookup_symbol_in_range (state_statics s) Hstatics sym allocated).
  - split; [ exact Htag | apply forallb_forall; exact Hfields ].
Qed.

Theorem integers_in_range : integers_in_range_statement.
Proof.
  intros s x kind i (_ & Hvars & _) Hlookup.
  apply int_of_value_in_range.
  exact (Hvars x _ (lookup_in _ _ _ Hlookup)).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The law is proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, which states and proves by itself the
   laws of the checks that the check of a term is made of. It is told
   one thing: to unfold the check of an arm of a [Switch] where it
   computes. The tactic does not use the law that it derives for that
   check at the arms of a renamed [Switch]; with the check unfolded it
   needs none. *)
#[local] Arguments arm_in_range arm /.

Theorem numbers_in_range_renamed : numbers_in_range_renamed_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Theorem in_range_runs : in_range_runs_statement.
Proof.
  split;
    [ vm_compute; reflexivity | exists 5%nat; eexists; vm_compute; reflexivity
     ].
Qed.

Theorem tag_out_of_range : tag_out_of_range_statement.
Proof.
  split;
    [ vm_compute; reflexivity | exists 5%nat; eexists; vm_compute; reflexivity
     ].
Qed.

Theorem constant_out_of_range : constant_out_of_range_statement.
Proof.
  split; [ vm_compute; reflexivity | exists 5%nat; vm_compute; reflexivity ].
Qed.
