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

(* What the rebuild with unboxing keeps of a unit of the fragment
   FlambdaReaperRegionsUnboxSpec.regions_local, as
   FlambdaReaperRegionsKeepsProofs.v has it of the rebuild with changes of
   calling convention: no symbol that the unit has not, no poison of the
   kind [Region] that the unit has not (the poison of a field of an
   unboxed block is of the kind of the field, which is not [Region]), and
   no handler with a parameter of that kind (the fields of an unboxed
   parameter are of the kinds of the fields). So the rebuilt unit is
   erasable and has no symbol at or above a base wherever the unit is and
   has not.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsUnboxProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaNameAbstractionSpec FlambdaRenameApartSpec FlambdaValueSpec
  FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaRegionsSpec
  FlambdaReaperRegionsSpec FlambdaReaperRegionsUnboxSpec
  FlambdaReaperRegionsRebuildProofs FlambdaReaperRegionsKeepsProofs
  FlambdaReaperRegionsUnboxRebuildProofs
  FlambdaReaperRegionsUnboxTraverseProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperUnboxingFuelSpec.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UF := FlambdaReaperUnboxingFuelSpec.

(* ======================================================================== *)
(* The kinds of the fields of the unboxed names                             *)
(* ======================================================================== *)

Lemma unboxed_fields_ukinds (F : facts) (T : tables)
    (cm : finite_map variable) (dominated : list (node * node))
    (to_unbox : list node) (next : positive) (x : node)
    (fields : unboxed_fields) :
  (∀ (w : node) (f : field) (b : node),
     (w, f, b) ∈ f_accessor F -> kind_not_region (field_kind f) = true) ->
  (x, fields)
    ∈ fst (unboxed_fields_of_names F T cm dominated to_unbox next) ->
  ukinds_ok (Unboxed fields) = true.
Proof.
  intros Hacc Hin.
  destruct (URP.unboxed_fields_of_names_in F T cm dominated to_unbox next x
              fields Hin) as (a & next' & ->).
  apply (mk_unboxed_fields_kinds F T cm dominated to_unbox Hacc).
  apply (get_fields_kinds F T Hacc).
Qed.

Lemma ukinds_fields (fields : unboxed_fields) :
  ukinds_ok (Unboxed fields) = true ->
  ∀ (f : field) (u : unboxed_field),
    (f, u) ∈ fields -> kind_not_region (field_kind f) = true.
Proof.
  induction fields as [ | [f0 u0] fields IH]; intros Hk f u Hin;
    [ destruct Hin | ].
  rewrite ukinds_ok_cons in Hk. apply andb_true_iff in Hk as [Hk Hr].
  apply andb_true_iff in Hk as [Hf _].
  destruct Hin as [Heq | Hin];
    [ injection Heq as <- <-; exact Hf | exact (IH Hr f u Hin) ].
Qed.

(* ======================================================================== *)
(* The rebuild with unboxing, case by case                                  *)
(* ======================================================================== *)

Section KeepsU.

Variable P : simple -> bool.
Variable Q : symbol -> bool.
Variable H : list flambda_kind -> bool.

Hypothesis HP_var : ∀ x : variable, P (Var x) = true.
Hypothesis HP_const :
  ∀ c : const, const_no_region_poison c = true -> P (Const c) = true.
Hypothesis HH_plain :
  ∀ l : list flambda_kind, forallb kind_not_region l = true -> H l = true.

Variables (F : facts) (sol : solution) (cc : node -> bool)
  (cds : finite_map code_dep) (b : bool) (ucids : list code_id)
  (codes : finite_map code_metadata) (ub : list (node * unboxed_fields))
  (next : positive).

Hypothesis Hkinds : ∀ (n : node) (fields : unboxed_fields),
  (n, fields) ∈ ub -> ukinds_ok (Unboxed fields) = true.

Abbreviation SO := (simples_ok P Q H).
Abbreviation solu := (solution_without_unboxed sol ub).
Abbreviation RU := (rebuild_expr_u false F sol cc cds b ucids codes ub next []
                      None).
Abbreviation RH := (rebuild_handler_u false F sol cc cds b ucids codes ub next
                      [] None).

Definition so_opt (o : option expr) : Prop :=
  match o with
  | Some e => SO e = true
  | None => True
  end.

Lemma so_of_opt (o : option expr) :
  so_opt o -> SO (match o with Some e => e | None => Invalid end) = true.
Proof. destruct o; [ exact (fun H => H) | intros _; reflexivity ]. Qed.

Lemma so_lets_simple (pairs : list (variable * variable * flambda_kind)) :
  ∀ body : expr,
    SO body = true ->
    SO (List.fold_left
          (fun body '(x, x', kind) => Let x kind (Simple (Var x')) body)
          pairs body)
    = true.
Proof.
  induction pairs as [ | [[x x'] k] pairs IH]; intros body Hb;
    [ exact Hb | ].
  cbn [List.fold_left]. apply IH.
  cbn [simples_ok simples_of_named forallb]. rewrite HP_var, Hb.
  reflexivity.
Qed.

Lemma so_bind_fields (u u' : unboxed_field) (body : expr) :
  SO body = true -> so_opt (bind_fields u u' body).
Proof.
  intros Hb. unfold bind_fields.
  destruct (unboxed_field_pairs Value u u');
    [ apply so_lets_simple, Hb | exact I ].
Qed.

Lemma so_bind_field_to_simple (u : unboxed_field) (kind : flambda_kind)
    (s : simple) (body : expr) :
  P s = true -> SO body = true -> so_opt (bind_field_to_simple u kind s body).
Proof.
  intros Hs Hb. destruct u; [ | exact I ].
  cbn [bind_field_to_simple so_opt simples_ok simples_of_named forallb].
  rewrite Hs, Hb. reflexivity.
Qed.

Lemma so_bind_block_fields (en : rebuild_env) (tag : Z) (args : list simple)
    (to_bind : unboxed_fields) (body : expr) :
  forallb P args = true -> ukinds_ok (Unboxed to_bind) = true ->
  SO body = true -> so_opt (bind_block_fields ub en tag args to_bind body).
Proof.
  intros Hargs Hk Hb. unfold bind_block_fields.
  pose proof (ukinds_fields to_bind Hk) as Hf. clear Hk.
  assert (Hacc : so_opt (Some body)) by exact Hb. revert Hacc.
  generalize (Some body) as acc.
  induction to_bind as [ | [f u] to_bind IH]; intros acc Hacc;
    [ exact Hacc | ].
  cbn [List.fold_left]. apply IH;
    [ intros f' u' Hin; apply (Hf f' u'); right; exact Hin | ].
  destruct acc as [b0 | ]; [ | exact I ]. cbn [so_opt] in Hacc.
  specialize (Hf f u (or_introl eq_refl)).
  destruct f as [nth kind | | | | | | | | ]; try exact I.
  - assert (Harg : P (match nth_error args nth with
                      | Some arg =>
                        if kind_eqb kind
                             (FlambdaReaperUnboxing.simple_kind en arg)
                        then arg
                        else Const (Poison kind)
                      | None => Const (Poison kind)
                      end)
                   = true).
    { destruct (nth_error args nth) as [a | ] eqn:Hn.
      - destruct (kind_eqb _ _).
        + rewrite forallb_forall in Hargs.
          exact (Hargs a (nth_error_In _ _ Hn)).
        + apply HP_const. exact Hf.
      - apply HP_const. exact Hf. }
    cbn beta iota. revert Harg.
    generalize (match nth_error args nth with
                | Some arg =>
                  if kind_eqb kind (FlambdaReaperUnboxing.simple_kind en arg)
                  then arg
                  else Const (Poison kind)
                | None => Const (Poison kind)
                end) as arg.
    intros arg Harg.
    destruct (unboxed_of ub arg);
      [ apply so_bind_fields, Hacc
      | apply so_bind_field_to_simple; [ exact Harg | exact Hacc ] ].
  - apply so_bind_field_to_simple; [ apply HP_const; reflexivity | exact Hacc ].
  - apply so_bind_field_to_simple; [ apply HP_const; reflexivity | exact Hacc ].
Qed.

Lemma so_bind_boxed_number_field (bn : boxable_number) (contents : simple)
    (to_bind : unboxed_fields) (body : expr) :
  P contents = true -> SO body = true ->
  so_opt (bind_boxed_number_field bn contents to_bind body).
Proof.
  intros Hc Hb. unfold bind_boxed_number_field.
  assert (Hacc : so_opt (Some body)) by exact Hb. revert Hacc.
  generalize (Some body) as acc.
  induction to_bind as [ | [f u] to_bind IH]; intros acc Hacc;
    [ exact Hacc | ].
  cbn [List.fold_left]. apply IH.
  destruct acc as [b0 | ]; [ | exact I ]. cbn [so_opt] in Hacc.
  destruct f; try exact I.
  destruct (Nat.eqb _ _); [ | exact I ].
  apply so_bind_field_to_simple; [ exact Hc | exact Hacc ].
Qed.

Lemma so_load_unboxed_field (f : field) (arg : simple)
    (to_bind : unboxed_fields) (body : expr) :
  SO body = true -> so_opt (load_unboxed_field ub [] f arg to_bind body).
Proof.
  intros Hb. unfold load_unboxed_field. destruct (unboxed_of ub arg).
  - destruct (Keyed.lookup field_eqb f _); [ apply so_bind_fields, Hb | ].
    exact I.
  - destruct arg; exact I.
Qed.

Lemma so_unboxed_binding (en : rebuild_env) (to_bind : unboxed_fields)
    (d : named) (body : expr) :
  forallb P (simples_of_named d) = true -> ukinds_ok (Unboxed to_bind) = true ->
  SO body = true -> SO (rebuild_unboxed_binding ub [] en to_bind d body) = true.
Proof.
  intros Hd Hk Hb. unfold rebuild_unboxed_binding. apply so_of_opt.
  destruct d as [s | p args | | | w arg | mf w arg | mf mt arg];
    try exact I; cbn [simples_of_named] in Hd.
  - destruct (unboxed_of ub s); [ apply so_bind_fields, Hb | exact I ].
  - destruct p as [p | p | p | p | p]; try exact I.
    + destruct p; try exact I; destruct args as [ | a [ | ]]; try exact I.
      * apply so_load_unboxed_field, Hb.
      * apply so_bind_boxed_number_field; [ | exact Hb ].
        cbn in Hd. apply andb_true_iff in Hd as [Hd _]. exact Hd.
    + destruct p as [bk | ak]; [ | exact I ].
      destruct bk; try exact I. apply so_bind_block_fields; assumption.
    + destruct p; try exact I. apply so_bind_block_fields; assumption.
    + destruct p; try exact I. exact Hb.
  - apply so_load_unboxed_field, Hb.
  - apply so_load_unboxed_field, Hb.
  - destruct (unboxed_of ub arg); [ apply so_bind_fields, Hb | exact I ].
Qed.

Lemma so_let_u (en : rebuild_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  forallb P (simples_of_named d) = true -> SO body = true ->
  SO (rebuild_let_u sol ub [] en x kind d body) = true.
Proof.
  intros Hd Hb. unfold rebuild_let_u.
  pose proof (named_ok P HP_var HP_const solu en x d Hd) as Hn.
  destruct (field_read d) as [[f arg] | ].
  - destruct (unboxed_of ub arg) as [af | ].
    + destruct (Keyed.lookup field_eqb f af) as [[v | n] | ]; try reflexivity.
      cbn [simples_ok simples_of_named forallb]. rewrite HP_var, Hb.
      reflexivity.
    + destruct arg; cbn [changed_of changed_of_node Keyed.lookup];
        cbn [simples_ok]; rewrite Hn, Hb; reflexivity.
  - cbn [simples_ok]. rewrite Hn, Hb. reflexivity.
Qed.

Lemma so_get_args (en : rebuild_env) (args : list simple) :
  ∀ (decisions : list decision) (args' : list simple),
    forallb P args = true ->
    get_args sol ub en decisions args = Some args' ->
    forallb P args' = true.
Proof.
  induction args as [ | a args IH]; intros ds args' Hargs E.
  - destruct ds; cbn in E; injection E as <-; reflexivity.
  - cbn [forallb] in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
    pose proof (P_rewrite P HP_var HP_const solu en a Ha) as Hr.
    destruct ds as [ | d ds]; cbn [get_args] in E.
    + destruct (get_args sol ub en [] args) as [rest | ] eqn:Er;
        [ | discriminate E ].
      cbn in E. injection E as <-. cbn [forallb].
      rewrite Hr. exact (IH [] rest Hargs Er).
    + destruct (get_args sol ub en ds args) as [rest | ] eqn:Er;
        [ | discriminate E ].
      pose proof (IH ds rest Hargs Er) as Hrest.
      destruct d as [[k | ] | fields | k].
      * injection E as <-. cbn [forallb]. rewrite Hr. exact Hrest.
      * injection E as <-. exact Hrest.
      * unfold unboxed_args in E.
        destruct (unboxed_of ub a); [ | discriminate E ].
        destruct (unboxed_field_pairs _ _ _) as [pairs | ];
          [ | discriminate E ].
        cbn in E. injection E as <-. rewrite forallb_app, Hrest, andb_true_r.
        rewrite map_map. apply forallb_forall. intros s Hin.
        apply in_map_iff in Hin as [[[x x'] kd] [<- _]]. apply HP_var.
      * injection E as <-. cbn [forallb]. rewrite Hr. exact Hrest.
Qed.

Lemma so_rewrite_apply_cont_u (en : unbox_env) (ac ac' : apply_cont) :
  rewrite_apply_cont_u sol ub en ac = Some ac' ->
  forallb P (ac_args ac) = true -> forallb P (ac_args ac') = true.
Proof.
  unfold rewrite_apply_cont_u. destruct (existsb _ _); [ discriminate | ].
  intros E Hac.
  destruct (lookup (ac_continuation ac) (ue_decisions en)) as [ds | ].
  - destruct (get_args sol ub (ue_env en) ds (ac_args ac)) as [args | ]
      eqn:Eg; [ | discriminate E ].
    injection E as <-. exact (so_get_args _ _ _ _ Hac Eg).
  - injection E as <-. cbn [ac_args]. apply forallb_forall.
    intros s' Hin. apply in_map_iff in Hin as [s [<- Hs]].
    apply (P_rewrite P HP_var HP_const). rewrite forallb_forall in Hac.
    exact (Hac s Hs).
Qed.

Lemma so_apply_cont_u (en : unbox_env) (ac : apply_cont) :
  forallb P (ac_args ac) = true ->
  SO (rebuild_apply_cont_u sol ub en ac) = true.
Proof.
  intros Hac. unfold rebuild_apply_cont_u.
  destruct (rewrite_apply_cont_u sol ub en ac) as [ac' | ] eqn:E;
    [ | reflexivity ].
  exact (so_rewrite_apply_cont_u en ac ac' E Hac).
Qed.

Lemma so_switch_u (en : unbox_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  P scrutinee = true ->
  forallb (fun '(_, ac) => forallb P (ac_args ac)) arms = true ->
  SO (rebuild_switch_u sol ub en scrutinee arms) = true.
Proof.
  intros Hs Harms. unfold rebuild_switch_u.
  match goal with
  |- context [ Keyed.is_empty ?l ] => destruct (Keyed.is_empty l)
  end; [ reflexivity | ].
  cbn [simples_ok]. rewrite (P_rewrite P HP_var HP_const solu _ _ Hs).
  cbn [andb].
  rewrite forallb_flat_map. apply forallb_forall. intros [d ac] Hin.
  apply in_compact_arms in Hin. rewrite forallb_forall in Harms.
  specialize (Harms _ Hin). cbn beta iota in Harms.
  destruct (rewrite_apply_cont_u sol ub en ac) as [ac' | ] eqn:E;
    [ | reflexivity ].
  cbn. rewrite (so_rewrite_apply_cont_u en ac ac' E Harms). reflexivity.
Qed.

Lemma hleaves : ∀ (n : node) (fields : unboxed_fields),
  (n, fields) ∈ ub ->
  forallb kind_not_region (map snd (fields_leaves fields)) = true.
Proof.
  intros n fields Hin. apply leaves_plain; [ reflexivity | ].
  exact (Hkinds n fields Hin).
Qed.

Lemma rebuild_u_simples :
  (∀ e : expr, ∀ en : unbox_env,
     closure_free e = true -> uenv_plain en ->
     SO e = true ->
     simples_ok (fun _ => true) (fun _ => true) no_region_kinds e = true ->
     SO (RU en e) = true)
  ∧ (∀ h : continuation_handler,
       ∀ (en : unbox_env) (dc : bool) (ds : list decision),
         handler_closure_free h = true ->
         handler_simples_ok P Q H h = true ->
         handler_simples_ok (fun _ => true) (fun _ => true) no_region_kinds h
         = true ->
         forallb decision_plain ds = true ->
         handler_simples_ok P Q H (RH en dc ds h) = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en Hcf Hpe Hso Hk.
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hcd Hcb].
    assert (Hset : named_is_set d = false)
      by (destruct d; try discriminate Hcd; reflexivity).
    cbn [simples_ok] in Hso, Hk. apply andb_true_iff in Hso as [Hd Hb].
    apply andb_true_iff in Hk as [_ Hkb].
    rewrite (ru_let_other F sol cc cds b ucids codes ub next en x kind d body
               (pending_plain en x d Hpe) Hset).
    assert (IHb : SO (RU (ubind_variables en [(x, kind)]) body) = true)
      by (apply IH; [ exact Hcb | apply uenv_plain_bind | exact Hb
                    | exact Hkb ]).
    destruct (is_dead_var sol x kind); [ reflexivity | ].
    destruct (is_var_used sol x kind); [ | exact IHb ].
    destruct (unboxed_of_node ub (Node_variable x)) as [tb | ] eqn:Hu.
    + apply so_unboxed_binding; [ exact Hd | | exact IHb ].
      apply (Hkinds (Node_variable x)).
      exact (keyed_lookup_in node_eqb URP.node_eqb_decides _ _ _ Hu).
    + apply so_let_u; [ exact Hd | exact IHb ].
  - (* Let_static *)
    intros bindings body _ IH en Hcf Hpe Hso Hk.
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hbc Hcb].
    cbn [simples_ok] in Hso, Hk. apply andb_true_iff in Hso as [Hbs Hb].
    apply andb_true_iff in Hk as [_ Hkb].
    rewrite ru_let_static. cbn [simples_ok].
    rewrite (IH _ Hcb (uenv_plain_bind _ _) Hb Hkb), andb_true_r.
    rewrite forallb_flat_map. rewrite forallb_forall in Hbc, Hbs |- *.
    intros bd Hin. specialize (Hbc bd Hin). specialize (Hbs bd Hin).
    destruct bd as [ | | sym [ | | tag fields | | | ]]; try discriminate Hbc.
    cbn [rebuild_binding_u static_set_repr].
    destruct (static_const_is_used sol sym _); [ | reflexivity ].
    change (binding_simples_ok P Q H
              (Bind_symbol sym
                 (rebuild_static_const solu (ue_env en) sym
                    (Static_const_block tag fields)))
            && true = true).
    rewrite andb_true_r.
    exact (static_const_ok P Q H HP_var HP_const solu _ _ _ Hbs).
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en Hcf Hpe Hso Hk.
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hhc Hcb].
    cbn [simples_ok] in Hso, Hk. apply andb_true_iff in Hso as [Hh Hb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    assert (Hhk : forallb kind_not_region (handler_kinds h) = true).
    { destruct h as [params md hbody]. cbn in Hkh |- *.
      apply andb_true_iff in Hkh as [Hkh _]. exact Hkh. }
    rewrite ru_let_cont.
    assert (IHb : ∀ bound : list (continuation * list decision),
              SO (RU (ubind_continuations en bound) body) = true)
      by (intros bound; apply IH;
          [ exact Hcb | apply uenv_plain_conts | exact Hb | exact Hkb ]).
    destruct (among k _); [ | apply IHb ].
    cbn [simples_ok]. rewrite IHb, andb_true_r.
    apply IHh; [ exact Hhc | exact Hh | exact Hkh | ].
    exact (handler_decisions_plain ub hleaves sol _ h Hhk).
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en Hcf Hpe Hso Hk.
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hhc Hcb].
    cbn [simples_ok] in Hso, Hk. apply andb_true_iff in Hso as [Hhs Hb].
    apply andb_true_iff in Hk as [Hkh Hkb].
    rewrite ru_let_cont_rec. cbn [simples_ok].
    rewrite (IH _ Hcb (uenv_plain_conts _ _) Hb Hkb), andb_true_r.
    rewrite forallb_forall in Hhc, Hhs, Hkh |- *.
    intros kh Hin. apply in_map_iff in Hin as [[k h] [<- Hin]].
    cbn beta iota.
    assert (Hhk : forallb kind_not_region (handler_kinds h) = true).
    { specialize (Hkh _ Hin). destruct h as [params md hbody].
      cbn in Hkh |- *. apply andb_true_iff in Hkh as [Hkh _]. exact Hkh. }
    apply (IHh k h Hin); [ exact (Hhc _ Hin) | exact (Hhs _ Hin)
                         | exact (Hkh _ Hin) | ].
    exact (handler_decisions_plain ub hleaves sol _ h Hhk).
  - (* Apply *)
    intros ap en Hcf. discriminate Hcf.
  - (* Apply_cont *)
    intros ac en _ _ Hso _. cbn [rebuild_expr_u].
    apply so_apply_cont_u. exact Hso.
  - (* Switch *)
    intros scrutinee arms en _ _ Hso _. cbn [rebuild_expr_u].
    cbn [simples_ok] in Hso. apply andb_true_iff in Hso as [Hs Harms].
    apply so_switch_u; assumption.
  - (* Invalid *)
    reflexivity.
  - (* Handler *)
    intros params md hbody IH en dc ds Hcf Hso Hk Hds.
    cbn [handler_simples_ok] in Hso, Hk.
    apply andb_true_iff in Hso as [_ Hb]. apply andb_true_iff in Hk as [Hk Hkb].
    rewrite rh_unfold. cbn [handler_simples_ok].
    rewrite HH_plain.
    2: { apply with_params_kinds_plain, params_of_decisions_plain;
           [ apply with_kinds_plain, Hk | exact Hds ]. }
    cbn [andb]. destruct (dc && _); [ reflexivity | ].
    apply IH; [ exact Hcf | apply uenv_plain_bind | exact Hb | exact Hkb ].
Qed.

End KeepsU.

(* ======================================================================== *)
(* The unit rebuilt                                                         *)
(* ======================================================================== *)

Lemma rebuild_unbox_keeps_simples (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) (P : simple -> bool)
    (Q : symbol -> bool) (H : list flambda_kind -> bool) :
  (∀ x : variable, P (Var x) = true) ->
  (∀ c : const, const_no_region_poison c = true -> P (Const c) = true) ->
  (∀ l : list flambda_kind, forallb kind_not_region l = true -> H l = true) ->
  regions_local e = true -> simples_ok P Q H e = true ->
  simples_ok P Q H
    (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
       dominated to_unbox b e)
  = true.
Proof.
  intros HP_var HP_const HH_plain Hloc Hso.
  unfold regions_local in Hloc. apply andb_true_iff in Hloc as [Hpl Hcf].
  unfold rebuild_unit_unbox, rebuild_unit_unbox_repr,
    rebuild_unit_unbox_repr_mode, unit_unboxed_fields.
  change (unit_changed_representations (fun _ => false)
            (traverse_unit flag u e) T dominated to_unbox [] e)
    with (@nil (node * closure_repr)).
  destruct (unboxed_fields_of_names (traverse_unit flag u e) T
              (bound_code_my_closures e) dominated to_unbox
              (fresh_name (names_of e)))
    as [ub next] eqn:Hub.
  assert (Hkinds : ∀ (n : node) (fields : unboxed_fields),
            (n, fields) ∈ ub -> ukinds_ok (Unboxed fields) = true).
  { intros n fields Hin.
    apply (unboxed_fields_ukinds (traverse_unit flag u e) T
             (bound_code_my_closures e) dominated to_unbox
             (fresh_name (names_of e)) n fields
             (closure_free_reads_plain flag u e Hcf)).
    rewrite Hub. exact Hin. }
  exact (proj1 (rebuild_u_simples P Q H HP_var HP_const HH_plain
                  (traverse_unit flag u e) (solution_of_tables T)
                  (fun n => mem_node n cannot_change) code_deps b
                  (static_names code_ids_bound e) (bound_code_metadata e) ub
                  next Hkinds)
           e empty_unbox_env Hcf (conj eq_refl eq_refl) Hso
           (proj1 calls_plain_handlers e []
              (proj1 regions_plain_calls e [] Hpl))).
Qed.

Lemma rebuild_unbox_erasable (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) :
  regions_local e = true -> erasable e empty_static_env = true ->
  erasable
    (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
       dominated to_unbox b e)
    empty_static_env
  = true.
Proof.
  intros Hloc He. pose proof Hloc as Hloc'. unfold regions_local in Hloc'.
  apply andb_true_iff in Hloc' as [Hpl Hcf].
  unfold erasable in He |- *.
  apply andb_true_iff in He as [He _]. apply andb_true_iff in He as [_ Hp].
  rewrite (proj1 calls_return_simples _ [] conts_plain_nil).
  2: { apply rebuild_unbox_keeps_simples; try reflexivity.
       - intros l Hl. exact Hl.
       - exact Hloc.
       - exact (proj1 calls_plain_handlers e []
                  (proj1 regions_plain_calls e [] Hpl)). }
  rewrite (proj1 no_region_poison_simples) in Hp |- *.
  rewrite (rebuild_unbox_keeps_simples flag u e T cannot_change code_deps
             dominated to_unbox b); try reflexivity; try assumption.
  intros c Hc. exact Hc.
Qed.

Lemma rebuild_unbox_fresh (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) (base : symbol) :
  regions_local e = true ->
  region_names_fresh base e empty_static_env = true ->
  region_names_fresh base
    (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
       dominated to_unbox b e)
    empty_static_env
  = true.
Proof.
  intros Hloc Hf. unfold region_names_fresh in Hf |- *.
  apply andb_true_iff in Hf as [Hf _]. apply andb_true_iff in Hf as [Hb _].
  rewrite (proj1 (expr_below_simples base)) in Hb |- *.
  rewrite (rebuild_unbox_keeps_simples flag u e T cannot_change code_deps
             dominated to_unbox b); try reflexivity; assumption.
Qed.
