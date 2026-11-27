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

(* The refinement of the local unboxing of
   FlambdaReaperUnboxingRefinesProofs.v, for any first fresh name of the
   variables of the fields above every name of the unit, and for a
   solution given apart from the tables that the unboxed fields are read
   off, which agrees with them on the variables, the sources and the
   usages, and may have more usages of other nodes
   ([rebuild_unbox_refines_rebuild_sol]). And the fragment of the local
   unboxing (FlambdaReaperUnboxingRefinesSpec.local_expr) reads the
   solution of the variables only ([local_expr_ext]), and of the unboxed
   fields their shapes only, which do not depend on the first fresh name
   ([local_expr_names]).

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsUnboxProofs.v. The proof of the refinement is that
   of FlambdaReaperUnboxingRefinesProofs.rebuild_unbox_with_refines_rebuild
   with the first fresh name and the solution left free. *)

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

(* ======================================================================== *)
(* A solution apart from the tables                                         *)
(* ======================================================================== *)

(* The solution [sol] has what the tables [T] have, and the same of the
   variables, of their sources and of the usages; it may have more
   usages of other nodes (FlambdaReaperRegionsProofs.consts_used gives
   the constants every usage). *)
Record coherent (T : tables) (sol : solution) : Prop := {
  co_has_usage : ∀ n : node,
    mem_node n (t_has_usage T) = true -> s_has_usage sol n = true;
  co_any_usage : ∀ n : node,
    mem_node n (t_any_usage T) = true -> s_any_usage sol n = true;
  co_has_usage_var : ∀ x : variable,
    s_has_usage sol (Node_variable x)
    = mem_node (Node_variable x) (t_has_usage T);
  co_any_usage_var : ∀ x : variable,
    s_any_usage sol (Node_variable x)
    = mem_node (Node_variable x) (t_any_usage T);
  co_has_source : ∀ n : node,
    s_has_source sol n = mem_node n (t_has_source T);
  co_usages : ∀ n n' : node,
    s_usages sol n n' = mem_pair n n' (t_usages T)
}.

Lemma coherent_tables (T : tables) : coherent T (solution_of_tables T).
Proof. split; intros; first [ assumption | reflexivity ]. Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Lemma rebuild_unbox_refines_rebuild_sol (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (sol : solution)
    (cannot_change : node -> bool) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (b : bool)
    (ff n0 next : positive) (unboxed : list (node * unboxed_fields)) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  coherent T sol ->
  (fresh_name (names_of e) <= n0)%positive ->
  unboxed_fields_of_names (traverse_unit flag u e) T
    (bound_code_my_closures e) dominated to_unbox n0
  = (unboxed, next) ->
  local_expr unboxed sol [] [] e = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_expr (traverse_unit flag u e) sol b
       (static_names code_ids_bound e) ff types empty_env e)
    (rebuild_expr_u false (traverse_unit flag u e) sol cannot_change
       code_deps b (static_names code_ids_bound e) (bound_code_metadata e)
       unboxed next [] types empty_unbox_env e).
Proof.
  intros Hslots Hfragment Hcl Hco Hn0 Hunboxed Hloc Hcheck.
  set (F := traverse_unit flag u e) in *.
  destruct (URP.unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
              dominated to_unbox n0)
    as (_ & Hnd & Hrange).
  rewrite Hunboxed in Hnd, Hrange. cbn [fst snd] in Hnd, Hrange.
  assert (Hdistinct : distinct (URP.field_variables unboxed) = true)
    by (apply distinct_NoDup, Hnd).
  assert (Hfresh : ∀ z : variable,
            z ∈ variables_of e -> among z (URP.field_variables unboxed)
                                  = false).
  { intros z Hz. destruct (among z _) eqn:Hamong; [ | reflexivity ].
    apply among_In in Hamong. destruct (Hrange z Hamong) as [Hle _].
    assert (Hnames : z ∈ names_of e)
      by (unfold names_of; apply in_or_app; left; exact Hz).
    apply URP.fold_max_le in Hnames. unfold fresh_name in Hn0. lia. }
  pose proof (URP.closure_free_reads_blocks flag u e
                (proj1 (URP.local_closure_free unboxed sol) e [] [] Hloc))
    as Hreads.
  apply (proj1 (URP.strong_local unboxed sol) e [] [] Hloc) in Hfresh.
  rename Hfresh into Hloc'.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages sol (Node_variable x) base = true
        ∧ s_has_usage sol w = true).
  { intros x fields f v Hx Hin. apply (URP.fields_of_in unboxed) in Hx.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox n0))
      by (rewrite Hunboxed; exact Hx).
    destruct (URP.unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _
                Hx')
      as (a & next' & ->).
    destruct (URP.mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f
                _ Hin)
      as (use & Huse).
    destruct (URP.get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (URP.add_usages_none F T _ Hreads) in Hbase.
    apply mem_node_In in Hbase.
    destruct (URP.get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split ].
    - rewrite (co_usages _ _ Hco). exact Hus.
    - destruct Hw as [Hw | Hw]; [ | exact (co_has_usage _ _ Hco _ Hw) ].
      exact (has_usage_any _ _ _ Hcl _ (co_any_usage _ _ Hco _ Hw)). }
  assert (Hreads_found :
    ∀ (x y : variable) (kind : flambda_kind) (f : field)
      (fields : unboxed_fields),
      fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
      g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
      is_dead_var sol x kind = false ->
      is_var_used sol x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hdead Hused Hkind.
    assert (Hsx : mem_node (Node_variable x) (t_has_source T) = true).
    { unfold is_dead_var in Hdead. rewrite <- (co_has_source _ _ Hco).
      destruct Hkind as [-> | (nk & ->)]; simpl in Hdead;
        apply negb_false_iff in Hdead; exact Hdead. }
    assert (Hsux : s_has_usage sol (Node_variable x) = true).
    { unfold is_var_used in Hused.
      destruct Hkind as [-> | (nk & ->)]; exact Hused. }
    assert (Hux : mem_node (Node_variable x) (t_has_usage T) = true)
      by (rewrite <- (co_has_usage_var _ _ Hco); exact Hsux).
    assert (Hhx : has_to_be_unboxed dominated to_unbox (Node_variable x)
                  = false).
    { destruct (has_to_be_unboxed dominated to_unbox (Node_variable x))
        eqn:Hh; [ | reflexivity ].
      unfold has_to_be_unboxed in Hh.
      apply existsb_exists in Hh as ([a x'] & Hin & Htest).
      apply andb_true_iff in Htest as [Hx' Ha].
      apply node_eqb_eq in Hx'. subst x'.
      pose proof (URP.unboxed_fields_of_names_bound F T
                    (bound_code_my_closures e) dominated to_unbox n0 a
                    (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb URP.node_eqb_decides _ _ Hx
               Hb). }
    apply (URP.fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox n0))
      by (rewrite Hunboxed; exact Hy).
    destruct (URP.unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _
                Hy')
      as (a & next' & Hdom & Ha & ->).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hblock : is_block_like_field f = true).
    { rewrite forallb_forall in Hreads. exact (Hreads _ Hacc). }
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hnau' : s_any_usage sol (Node_variable y) = false)
      by (rewrite (co_any_usage_var _ _ Hco); exact Hnau).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true).
    { rewrite <- (co_usages _ _ Hco).
      exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau'). }
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true).
    { rewrite <- (co_has_usage_var _ _ Hco).
      exact (has_usage_accessor _ _ _ Hcl _ _ _ Hsux Hacc). }
    assert (Huses : mem_node (Node_variable y)
                      (add_usages_through_function_slots F T
                         (bound_code_my_closures e) true
                         (get_direct_usages T [Node_variable y]))
                    = true).
    { rewrite (URP.add_usages_none F T _ Hreads). apply mem_node_In.
      apply URP.get_direct_usages_self; assumption. }
    destruct (URP.get_fields_found F T _ _ _ _ Hacc Huses Hfs Hux)
      as [Hex Hall].
    apply (URP.mk_unboxed_fields_lookup F T _ dominated to_unbox _ a f
             Hdata).
    - intros Hno. rewrite Hno in Hsrc. discriminate Hsrc.
    - exact Hex.
    - intros use Hin.
      destruct (Hall use Hin) as [-> | (flow_to & -> & Hw)];
        [ left; reflexivity | right ].
      exists flow_to. split; [ reflexivity | ].
      destruct (forallb _ flow_to) eqn:Hft; [ | reflexivity ].
      rewrite forallb_forall in Hft. rewrite (Hft _ Hw) in Hhx.
      discriminate Hhx. }
  destruct (proj1 (URP.rebuild_ub F sol cannot_change code_deps b
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     unboxed ff next types noted deps cds (unit_kept e)
                     (unit_codes e) (static_names symbols_of_binding e)
                     is_local Hcl Hslots Hfields_read Hreads_found)
              e empty_unbox_env [] (unit_conts u) URP.uinv_empty
              (Forall2_nil _) Hloc' Hedges)
    as (Hub & _ & _).
  exact (URP.ub_refines unboxed Hdistinct _ _ Hub).
Qed.
