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

(* The fragment of the local unboxing
   (FlambdaReaperUnboxingRefinesSpec.local_expr) reads the solution of the
   variables only ([local_expr_ext]), and of the unboxed fields their
   shapes only, which do not depend on the first fresh name
   ([local_expr_names]). With FlambdaReaperUnboxingRefinesProofs
   .rebuild_unbox_refines_rebuild_sol, the refinement of the local
   unboxing for a solution given apart from the tables and any first
   fresh name, it gives the local unboxing on the erased unit.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsUnboxProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperFlowSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaUnusedStaticsSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperRegionsRebuildProofs FlambdaReaperUnboxingFuelSpec
  FlambdaReaperUnboxingFuelProofs
  FlambdaReaperUnboxingRefinesProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UF := FlambdaReaperUnboxingFuelSpec.
Module UFP := FlambdaReaperUnboxingFuelProofs.

(* ======================================================================== *)
(* The fragment reads the solution of the variables only                    *)
(* ======================================================================== *)

Section SolExt.

Variable unboxed : list (node * unboxed_fields).
Variables sol sol' : solution.
Hypothesis Hvar : ∀ x : variable,
  s_has_usage sol' (Node_variable x) = s_has_usage sol (Node_variable x).

Lemma var_used_ext (x : variable) (kind : flambda_kind) :
  is_var_used sol' x kind = is_var_used sol x kind.
Proof. destruct kind; cbn; try reflexivity; apply Hvar. Qed.

Lemma handler_mask_ext (b : bool) (h : continuation_handler) :
  handler_mask sol' b h = handler_mask sol b h.
Proof.
  destruct h as [params md body]. cbn [handler_mask]. unfold params_to_keep.
  apply map_ext. intros [x k]. rewrite var_used_ext. reflexivity.
Qed.

Lemma unboxed_arg_ok_ext (kinds : finite_map flambda_kind) (keep : bool)
    (fields : unboxed_fields) (a : simple) :
  unboxed_arg_ok unboxed sol' kinds keep fields a
  = unboxed_arg_ok unboxed sol kinds keep fields a.
Proof.
  unfold unboxed_arg_ok. destruct a as [y | | ]; try reflexivity.
  destruct (fields_of unboxed y); [ rewrite var_used_ext | ]; reflexivity.
Qed.

Lemma unboxed_binding_ok_ext (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named)
    (fields : unboxed_fields) :
  unboxed_binding_ok unboxed sol' kinds x kind d fields
  = unboxed_binding_ok unboxed sol kinds x kind d fields.
Proof.
  unfold unboxed_binding_ok.
  destruct kind; try reflexivity.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  - destruct s; try reflexivity. rewrite unboxed_arg_ok_ext. reflexivity.
  - destruct p as [p | p | p | p | p]; try reflexivity.
    destruct p as [bk | ]; try reflexivity. destruct bk; try reflexivity.
    induction fields as [ | [f u] fields IH]; [ reflexivity | ].
    cbn [forallb]. rewrite IH. f_equal.
    destruct f as [i k | | | | | | | | ]; try reflexivity.
    destruct k; try reflexivity. destruct u as [v | nested];
      [ reflexivity | ].
    destruct (nth_error args i) as [[b | | ] | ]; try reflexivity.
    rewrite unboxed_arg_ok_ext. reflexivity.
Qed.

Lemma args_ok_ext (kinds : finite_map flambda_kind)
    (params : list (variable * bool)) (args : list simple) :
  args_ok unboxed sol' kinds params args
  = args_ok unboxed sol kinds params args.
Proof.
  revert args. induction params as [ | [p keep] params IH];
    intros [ | a args]; try reflexivity.
  cbn [args_ok]. rewrite IH.
  destruct (fields_of unboxed p); [ rewrite unboxed_arg_ok_ext | ];
    reflexivity.
Qed.

Lemma apply_cont_ok_ext (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) :
  apply_cont_ok unboxed sol' kinds conts ac
  = apply_cont_ok unboxed sol kinds conts ac.
Proof.
  unfold apply_cont_ok. destruct (lookup _ conts); [ | reflexivity ].
  apply args_ok_ext.
Qed.

Lemma local_expr_ext :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     local_expr unboxed sol' kinds conts e
     = local_expr unboxed sol kinds conts e)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       local_handler unboxed sol' kinds conts mask h
       = local_handler unboxed sol kinds conts mask h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts. cbn [local_expr].
    rewrite IH. destruct (fields_of unboxed x);
      [ rewrite unboxed_binding_ok_ext | ]; reflexivity.
  - intros bindings body _ IH kinds conts. cbn [local_expr].
    rewrite IH. reflexivity.
  - intros k h body IHh IH kinds conts. cbn [local_expr].
    rewrite !handler_mask_ext, IHh, IH. reflexivity.
  - intros handlers body IHh IH kinds conts. cbn [local_expr].
    assert (Hm : map (fun '(k, h) =>
                        (k, combine (handler_variables h)
                              (handler_mask sol' false h))) handlers
                 = map (fun '(k, h) =>
                          (k, combine (handler_variables h)
                                (handler_mask sol false h))) handlers).
    { apply map_ext. intros [k h]. rewrite handler_mask_ext. reflexivity. }
    rewrite Hm, IH. f_equal.
    apply FlambdaReaperRegionsRebuildProofs.forallb_ext_in.
    intros [k h] Hin. rewrite handler_mask_ext. exact (IHh k h Hin _ _ _).
  - reflexivity.
  - intros ac kinds conts. cbn [local_expr]. apply apply_cont_ok_ext.
  - intros scrutinee arms kinds conts. cbn [local_expr]. f_equal.
    apply FlambdaReaperRegionsRebuildProofs.forallb_ext_in.
    intros [i ac] _. apply apply_cont_ok_ext.
  - reflexivity.
  - intros params md handler IH kinds conts mask. cbn [local_handler].
    rewrite IH. reflexivity.
Qed.

End SolExt.

(* ======================================================================== *)
(* The fragment reads the shapes of the unboxed fields only                 *)
(* ======================================================================== *)

(* An unboxing with every variable of its fields renamed to 1. *)
Fixpoint ushape (u : unboxed_field) : unboxed_field :=
  match u with
  | Not_unboxed _ => Not_unboxed xH
  | Unboxed fields =>
    Unboxed
      ((fix go (l : unboxed_fields) : unboxed_fields :=
          match l with
          | [] => []
          | (f, u) :: l => (f, ushape u) :: go l
          end) fields)
  end.

Definition fshape (fields : unboxed_fields) : unboxed_fields :=
  map (fun '(f, u) => (f, ushape u)) fields.

Lemma ushape_unboxed (fields : unboxed_fields) :
  ushape (Unboxed fields) = Unboxed (fshape fields).
Proof.
  cbn [ushape]. f_equal. unfold fshape.
  induction fields as [ | [f u] fields IH]; [ reflexivity | ].
  cbn [map]. rewrite <- IH. reflexivity.
Qed.

Definition shapes (l : list (node * unboxed_fields))
    : list (node * unboxed_fields) :=
  map (fun '(n, fields) => (n, fshape fields)) l.

Lemma lookup_map_snd {K A B : Type} (eqb : K -> K -> bool) (g : A -> B)
    (k : K) (l : list (K * A)) :
  Keyed.lookup eqb k (map (fun '(n, a) => (n, g a)) l)
  = option_map g (Keyed.lookup eqb k l).
Proof.
  induction l as [ | [n a] l IH]; cbn; [ reflexivity | ].
  destruct (eqb k n); [ reflexivity | exact IH ].
Qed.

Lemma lookup_fshape (f : field) (fields : unboxed_fields) :
  Keyed.lookup field_eqb f (fshape fields)
  = option_map ushape (Keyed.lookup field_eqb f fields).
Proof. apply lookup_map_snd. Qed.

Definition is_some {A : Type} (o : option A) : bool :=
  match o with
  | Some _ => true
  | None => false
  end.

Lemma is_some_app {A : Type} (o o' : option (list A)) :
  is_some (match o, o' with
           | Some p, Some q => Some (p ++ q)
           | _, _ => None
           end)
  = is_some o && is_some o'.
Proof. destruct o, o'; reflexivity. Qed.

Lemma pairs_shape :
  ∀ (u : unboxed_field) (kind : flambda_kind) (u' : unboxed_field),
    is_some (unboxed_field_pairs kind (ushape u) (ushape u'))
    = is_some (unboxed_field_pairs kind u u').
Proof.
  refine (URP.unboxed_field_rec _ _ _).
  - intros v kind [v' | fields']; [ reflexivity | ].
    rewrite ushape_unboxed. reflexivity.
  - intros fields IH kind [v' | fields'];
      [ rewrite ushape_unboxed; reflexivity | ].
    rewrite !ushape_unboxed. cbn [unboxed_field_pairs]. unfold fshape.
    induction fields as [ | [f u] fields IHl]; [ reflexivity | ].
    cbn [map]. rewrite lookup_map_snd.
    destruct (Keyed.lookup field_eqb f fields') as [u0 | ];
      [ | reflexivity ].
    cbn [option_map]. rewrite !is_some_app.
    rewrite (IH f u (or_introl eq_refl)).
    f_equal. apply IHl. intros f' u' Hin. apply (IH f' u'). right. exact Hin.
Qed.

Lemma fields_alike_shapes (fields fields' : unboxed_fields) :
  fields_alike (fshape fields) (fshape fields') = fields_alike fields fields'.
Proof.
  unfold fields_alike. rewrite <- !ushape_unboxed.
  pose proof (pairs_shape (Unboxed fields) Value (Unboxed fields')) as H.
  destruct (unboxed_field_pairs Value (ushape (Unboxed fields))
              (ushape (Unboxed fields'))),
    (unboxed_field_pairs Value (Unboxed fields) (Unboxed fields'));
    try reflexivity; discriminate H.
Qed.

Section Shapes.

Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

Abbreviation S := (shapes unboxed).

Lemma fields_of_shapes (x : variable) :
  fields_of S x = option_map fshape (fields_of unboxed x).
Proof. apply lookup_map_snd. Qed.

Lemma is_bound_shapes (n : node) :
  Keyed.is_bound node_eqb n S = Keyed.is_bound node_eqb n unboxed.
Proof.
  unfold Keyed.is_bound, shapes. rewrite lookup_map_snd.
  destruct (Keyed.lookup _ _ _); reflexivity.
Qed.

Lemma plain_simple_shapes (s : simple) :
  plain_simple S s = plain_simple unboxed s.
Proof.
  destruct s; cbn [plain_simple]; unfold plain_variable;
    rewrite ?is_bound_shapes; reflexivity.
Qed.

Lemma plain_variable_shapes (x : variable) :
  plain_variable S x = plain_variable unboxed x.
Proof. unfold plain_variable. rewrite is_bound_shapes. reflexivity. Qed.

Lemma forallb_plain_shapes (l : list simple) :
  forallb (plain_simple S) l = forallb (plain_simple unboxed) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ].
  cbn [forallb]. rewrite IH, plain_simple_shapes. reflexivity.
Qed.

Lemma unboxed_arg_ok_shapes (kinds : finite_map flambda_kind) (keep : bool)
    (fields : unboxed_fields) (a : simple) :
  unboxed_arg_ok S sol kinds keep (fshape fields) a
  = unboxed_arg_ok unboxed sol kinds keep fields a.
Proof.
  unfold unboxed_arg_ok. destruct a as [y | | ]; try reflexivity.
  rewrite fields_of_shapes.
  destruct (fields_of unboxed y); [ | reflexivity ].
  cbn [option_map]. rewrite fields_alike_shapes. reflexivity.
Qed.

Lemma unboxed_binding_ok_shapes (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named)
    (fields : unboxed_fields) :
  unboxed_binding_ok S sol kinds x kind d (fshape fields)
  = unboxed_binding_ok unboxed sol kinds x kind d fields.
Proof.
  unfold unboxed_binding_ok.
  destruct kind; try reflexivity.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  - destruct s; try reflexivity. rewrite unboxed_arg_ok_shapes. reflexivity.
  - destruct p as [p | p | p | p | p]; try reflexivity.
    + destruct p; try reflexivity.
      * destruct args as [ | [y | | ] [ | ]]; try reflexivity.
        rewrite fields_of_shapes. destruct (fields_of unboxed y);
          [ | reflexivity ].
        cbn [option_map]. rewrite lookup_fshape.
        destruct (Keyed.lookup _ _ _) as [[v | nested] | ]; try reflexivity.
        cbn [option_map]. rewrite ushape_unboxed, fields_alike_shapes.
        reflexivity.
      * destruct args as [ | c [ | ]]; try reflexivity.
        rewrite plain_simple_shapes. f_equal.
        induction fields as [ | [f u] fields IH]; [ reflexivity | ].
        cbn [fshape map forallb]. rewrite <- IH.
        destruct u; [ reflexivity | rewrite ushape_unboxed; reflexivity ].
    + destruct p as [bk | ]; try reflexivity. destruct bk; try reflexivity.
      induction fields as [ | [f u] fields IH]; [ reflexivity | ].
      cbn [fshape map forallb]. rewrite <- IH. f_equal.
      destruct f as [i k | | | | | | | | ]; try reflexivity.
      destruct k; try reflexivity.
      destruct u as [v | nested].
      * cbn [ushape]. destruct (nth_error args i); [ | reflexivity ].
        rewrite plain_simple_shapes. reflexivity.
      * rewrite ushape_unboxed.
        destruct (nth_error args i) as [[b | | ] | ]; try reflexivity.
        rewrite unboxed_arg_ok_shapes. reflexivity.
Qed.

Lemma unboxed_read_shapes (d : named) :
  is_some (unboxed_read S d) = is_some (unboxed_read unboxed d).
Proof.
  unfold unboxed_read.
  destruct d as [ | p args | | | | | ]; try reflexivity.
  destruct p as [p | | | | ]; try reflexivity.
  destruct p; try reflexivity;
    destruct args as [ | [y | | ] [ | ]]; try reflexivity;
    destruct (field_read _) as [[f s] | ]; try reflexivity;
    rewrite fields_of_shapes; destruct (fields_of unboxed y); reflexivity.
Qed.

Lemma plain_binding_ok_shapes (x : variable) (kind : flambda_kind)
    (d : named) :
  plain_binding_ok S x kind d = plain_binding_ok unboxed x kind d.
Proof.
  unfold plain_binding_ok. pose proof (unboxed_read_shapes d) as H.
  destruct (unboxed_read S d), (unboxed_read unboxed d);
    try discriminate H; [ reflexivity | ].
  destruct d; try reflexivity.
  - apply plain_simple_shapes.
  - induction args as [ | a args IH]; [ reflexivity | ].
    cbn [forallb]. rewrite IH, plain_simple_shapes. reflexivity.
Qed.

Lemma args_ok_shapes (kinds : finite_map flambda_kind)
    (params : list (variable * bool)) (args : list simple) :
  args_ok S sol kinds params args = args_ok unboxed sol kinds params args.
Proof.
  revert args. induction params as [ | [p keep] params IH];
    intros [ | a args]; try reflexivity.
  - cbn [args_ok]. apply forallb_plain_shapes.
  - cbn [args_ok]. rewrite IH, fields_of_shapes.
    destruct (fields_of unboxed p); cbn [option_map];
      [ rewrite unboxed_arg_ok_shapes | rewrite plain_simple_shapes ];
      reflexivity.
Qed.

Lemma apply_cont_ok_shapes (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) :
  apply_cont_ok S sol kinds conts ac = apply_cont_ok unboxed sol kinds conts ac.
Proof.
  unfold apply_cont_ok. destruct (lookup _ conts); [ apply args_ok_shapes | ].
  induction (ac_args ac) as [ | a args IH]; [ reflexivity | ].
  cbn [forallb]. rewrite IH, plain_simple_shapes. reflexivity.
Qed.

Lemma param_ok_shapes (p : variable) (kind : flambda_kind) (keep : bool) :
  param_ok S p kind keep = param_ok unboxed p kind keep.
Proof.
  unfold param_ok. rewrite fields_of_shapes.
  destruct (fields_of unboxed p) as [[ | ] | ]; reflexivity.
Qed.

Lemma local_expr_shapes :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     local_expr S sol kinds conts e = local_expr unboxed sol kinds conts e)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       local_handler S sol kinds conts mask h
       = local_handler unboxed sol kinds conts mask h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts. cbn [local_expr].
    rewrite IH, fields_of_shapes.
    destruct (fields_of unboxed x); cbn [option_map];
      [ rewrite unboxed_binding_ok_shapes | rewrite plain_binding_ok_shapes ];
      reflexivity.
  - intros bindings body _ IH kinds conts. cbn [local_expr].
    rewrite IH. f_equal. unfold local_binding.
    apply FlambdaReaperRegionsRebuildProofs.forallb_ext_in.
    intros [ | | s [ | | tag fields | | | ]] _; try reflexivity.
    induction fields as [ | a fields IHf]; [ reflexivity | ].
    cbn [forallb]. rewrite IHf, plain_simple_shapes. reflexivity.
  - intros k h body IHh IH kinds conts. cbn [local_expr].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH kinds conts. cbn [local_expr].
    rewrite IH. f_equal.
    apply FlambdaReaperRegionsRebuildProofs.forallb_ext_in.
    intros [k h] Hin. exact (IHh k h Hin _ _ _).
  - reflexivity.
  - intros ac kinds conts. cbn [local_expr]. apply apply_cont_ok_shapes.
  - intros scrutinee arms kinds conts. cbn [local_expr].
    rewrite plain_simple_shapes. f_equal.
    apply FlambdaReaperRegionsRebuildProofs.forallb_ext_in.
    intros [i ac] _. apply apply_cont_ok_shapes.
  - reflexivity.
  - intros params md handler IH kinds conts mask. cbn [local_handler].
    rewrite IH. f_equal. f_equal.
    apply FlambdaReaperRegionsRebuildProofs.forallb_ext_in.
    intros [[p kind] keep] _. apply param_ok_shapes.
Qed.

End Shapes.

(* ======================================================================== *)
(* The shapes do not depend on the first fresh name                         *)
(* ======================================================================== *)

Section Names.

Variables (F : facts) (T : tables) (cm : finite_map variable)
  (dominated : list (node * node)) (to_unbox : list node).

Lemma field_step_shape (fuel : nat) :
  (∀ fuel' : nat, fuel = S fuel' ->
     ∀ (block : node) (fields : list (field * field_use)) (n n' : positive),
       fshape (fst (mk_unboxed_fields F T cm dominated to_unbox fuel' block
                      fields n))
       = fshape (fst (mk_unboxed_fields F T cm dominated to_unbox fuel' block
                        fields n'))) ->
  ∀ (block : node) (f : field) (use : field_use) (n n' : positive),
    option_map ushape
      (fst (URP.field_step F T cm dominated to_unbox fuel block f use n))
    = option_map ushape
        (fst (URP.field_step F T cm dominated to_unbox fuel block f use n')).
Proof.
  intros Hnest block f use n n'. unfold URP.field_step.
  destruct (negb (is_data_field f)); [ reflexivity | ].
  destruct (get_single_field_source F T block f) as [ | source | ];
    [ reflexivity | | ];
    destruct use as [ | flow_to ]; try reflexivity;
    destruct (forallb _ flow_to); try reflexivity.
  destruct fuel as [ | fuel']; [ reflexivity | ].
  pose proof (Hnest fuel' eq_refl source
                (get_fields F T
                   (add_usages_through_function_slots F T cm true
                      (get_direct_usages T flow_to)))
                n n') as H.
  destruct (mk_unboxed_fields _ _ _ _ _ fuel' source _ n) as [nested m].
  destruct (mk_unboxed_fields _ _ _ _ _ fuel' source _ n')
    as [nested' m'].
  cbn [fst option_map] in H |- *. rewrite !ushape_unboxed, H. reflexivity.
Qed.

Lemma mk_unboxed_fields_shape (fuel : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (n n' : positive),
    fshape (fst (mk_unboxed_fields F T cm dominated to_unbox fuel block
                   fields n))
    = fshape (fst (mk_unboxed_fields F T cm dominated to_unbox fuel block
                     fields n')).
Proof.
  induction fuel as [ | fuel IHf]; intros block fields;
    [ assert (Hn : ∀ fuel', 0 = S fuel' ->
                 ∀ block fields n n',
                   fshape (fst (mk_unboxed_fields F T cm dominated to_unbox
                                  fuel' block fields n))
                   = fshape (fst (mk_unboxed_fields F T cm dominated
                                    to_unbox fuel' block fields n')))
        by (intros ? H; discriminate H)
    | assert (Hn : ∀ fuel', S fuel = S fuel' ->
                 ∀ block fields n n',
                   fshape (fst (mk_unboxed_fields F T cm dominated to_unbox
                                  fuel' block fields n))
                   = fshape (fst (mk_unboxed_fields F T cm dominated
                                    to_unbox fuel' block fields n')))
        by (intros ? H; injection H as <-; exact IHf) ];
    induction fields as [ | [f use] fields IH]; intros n n';
    rewrite ?URP.mk_unboxed_fields_nil; try reflexivity;
    rewrite !URP.mk_unboxed_fields_cons;
    pose proof (field_step_shape _ Hn block f use n n') as Hs;
    destruct (URP.field_step _ _ _ _ _ _ block f use n) as [u m];
    destruct (URP.field_step _ _ _ _ _ _ block f use n') as [u' m'];
    pose proof (IH m m') as Hr;
    destruct (mk_unboxed_fields _ _ _ _ _ _ block fields m) as [rest k];
    destruct (mk_unboxed_fields _ _ _ _ _ _ block fields m')
      as [rest' k'];
    cbn [fst option_map] in Hs, Hr |- *;
    (destruct u as [u | ], u' as [u' | ];
     try discriminate Hs; [ injection Hs as Hs | exact Hr ]);
    unfold fshape in Hr |- *; cbn [map]; rewrite Hs, Hr; reflexivity.
Qed.

Lemma existsb_keys_shapes (x : node) (l : list (node * unboxed_fields)) :
  existsb (fun '(y, _) => node_eqb x y) (shapes l)
  = existsb (fun '(y, _) => node_eqb x y) l.
Proof.
  induction l as [ | [y fs] l IH]; [ reflexivity | ].
  cbn [shapes map existsb]. rewrite <- IH. reflexivity.
Qed.

Lemma unboxed_fields_of_names_shapes (n n' : positive) :
  shapes (fst (unboxed_fields_of_names F T cm dominated to_unbox n))
  = shapes (fst (unboxed_fields_of_names F T cm dominated to_unbox n')).
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc acc' : list (node * unboxed_fields)) (m m' : positive),
    shapes acc = shapes acc' ->
    shapes (fst (List.fold_left
                   (fun '(acc, next) '(a, x) =>
                      if mem_node a to_unbox
                         && negb (existsb (fun '(y, _) => node_eqb x y) acc)
                      then
                        let '(fields, next) :=
                          name_unboxed_fields F T cm dominated to_unbox a x
                            next
                        in
                        (acc ++ [(x, fields)], next)
                      else (acc, next))
                   l (acc, m)))
    = shapes (fst (List.fold_left
                     (fun '(acc, next) '(a, x) =>
                        if mem_node a to_unbox
                           && negb (existsb (fun '(y, _) => node_eqb x y)
                                      acc)
                        then
                          let '(fields, next) :=
                            name_unboxed_fields F T cm dominated to_unbox a x
                              next
                          in
                          (acc ++ [(x, fields)], next)
                        else (acc, next))
                     l (acc', m')))).
  { induction l as [ | [a x] l IHl]; intros acc acc' m m' Hacc;
      [ exact Hacc | ].
    cbn [List.fold_left].
    rewrite <- (existsb_keys_shapes x acc), <- (existsb_keys_shapes x acc'),
      Hacc.
    destruct (mem_node a to_unbox
              && negb (existsb (fun '(y, _) => node_eqb x y) (shapes acc')));
      [ | apply IHl, Hacc ].
    rewrite !UFP.name_unboxed_fields_depth.
    pose proof (mk_unboxed_fields_shape (UF.unboxing_depth T) a
                  (get_fields F T
                     (add_usages_through_function_slots F T cm true
                        (get_direct_usages T [x])))
                  m m') as Hs.
    destruct (mk_unboxed_fields _ _ _ _ _ _ a _ m) as [fs k].
    destruct (mk_unboxed_fields _ _ _ _ _ _ a _ m') as [fs' k'].
    cbn [fst] in Hs. apply IHl.
    unfold shapes. rewrite !map_app. fold (shapes acc) (shapes acc').
    rewrite Hacc. cbn [map]. rewrite Hs. reflexivity. }
  apply Hgen. reflexivity.
Qed.

End Names.

(* The fragment of the local unboxing does not depend on the first fresh
   name of the variables of the fields. *)
Lemma local_expr_names (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (sol : solution)
    (n n' : positive) (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (e : expr) :
  local_expr (fst (unboxed_fields_of_names F T cm dominated to_unbox n)) sol
    kinds conts e
  = local_expr (fst (unboxed_fields_of_names F T cm dominated to_unbox n'))
      sol kinds conts e.
Proof.
  rewrite <- (proj1 (local_expr_shapes _ sol) e kinds conts).
  rewrite <- (proj1 (local_expr_shapes
                       (fst (unboxed_fields_of_names F T cm dominated
                               to_unbox n')) sol) e kinds conts).
  rewrite (unboxed_fields_of_names_shapes F T cm dominated to_unbox n n').
  reflexivity.
Qed.
