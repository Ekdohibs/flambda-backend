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

(* Proofs of the statements of FlambdaFreeNamesSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaBindersProofs
  FlambdaNominalSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaNormaliseSpec FlambdaFreeNamesSpec.
Import ListNotations.

(* [all_among] is the test, as in FlambdaBindersProofs.v. *)
#[local] Abbreviation all_among := FlambdaMapSpec.all_among.

(* ======================================================================== *)
(* Lists and sets                                                           *)
(* ======================================================================== *)

Lemma among_singleton (name name' : positive) :
  among name [name'] = true <-> name = name'.
Proof.
  rewrite among_In. simpl. split.
  - intros [-> | []]. reflexivity.
  - intros ->. left. reflexivity.
Qed.

Lemma not_among_in (name : positive) (names : list positive) :
  name ∉ names <-> among name names = false.
Proof.
  split.
  - intros Hnot. destruct (among name names) eqn:Hname; [ | reflexivity ].
    apply among_In in Hname. contradiction.
  - intros Hname Hin. apply among_In in Hin. congruence.
Qed.

Lemma not_among_singleton (name name' : positive) :
  name <> name' <-> among name [name'] = false.
Proof.
  rewrite <- not_among_in. simpl. split.
  - intros Hne [Heq | []]. congruence.
  - intros Hnot Heq. apply Hnot. left. congruence.
Qed.

Lemma among_flat_map {A : Type} (f : A -> list positive) (l : list A)
    (name : positive) :
  among name (flat_map f l) = true
  <-> Exists (fun a => among name (f a) = true) l.
Proof.
  rewrite among_In, in_flat_map, Exists_exists.
  split; intros [a [Hin Ha]]; exists a; (split; [ exact Hin | ]);
    apply among_In, Ha.
Qed.

Lemma Exists_equiv {A : Type} (P Q : A -> Prop) (l : list A) :
  (∀ a : A, P a <-> Q a) -> (Exists P l <-> Exists Q l).
Proof.
  intros Hequiv. rewrite !Exists_exists.
  split; intros [a [Hin Ha]]; exists a; (split; [ exact Hin | ]);
    apply Hequiv, Ha.
Qed.

Lemma all_among_cons (name : positive) (names allowed : list positive) :
  all_among (name :: names) allowed
  = among name allowed && all_among names allowed.
Proof. reflexivity. Qed.

(* The free variables of the static constants of a group: what
   [closed_code_free_names] says the free variables of the group are. *)
Definition free_variables_of_constants (bindings : list static_binding)
    : list variable :=
  flat_map
    (fun binding =>
       match binding with
       | Bind_symbol _ const => free_variables_of_static_const const
       | Bind_code _ _ | Bind_deleted_code _ => []
       end)
    bindings.

(* ======================================================================== *)
(* The relation on variables                                                *)
(* ======================================================================== *)

Lemma var_free_in_simple_among (x : variable) (s : simple) :
  var_free_in_simple x s <-> among x (free_variables_of_simple s) = true.
Proof.
  destruct s as [y | sym | c].
  - symmetry. apply among_singleton.
  - simpl. split; [ intros [] | discriminate ].
  - simpl. split; [ intros [] | discriminate ].
Qed.

Lemma var_free_in_simples_among (x : variable) (args : list simple) :
  var_free_in_simples x args
  <-> among x (free_variables_of_simples args) = true.
Proof.
  unfold var_free_in_simples, free_variables_of_simples.
  rewrite among_flat_map. apply Exists_equiv. intros s.
  apply var_free_in_simple_among.
Qed.

Lemma var_free_in_value_slots_among (x : variable)
    (value_slots : list (value_slot * simple)) :
  Exists (fun slot => var_free_in_simple x (snd slot)) value_slots
  <-> among x (free_variables_of_value_slots value_slots) = true.
Proof.
  unfold free_variables_of_value_slots.
  rewrite among_flat_map. apply Exists_equiv. intros slot.
  apply var_free_in_simple_among.
Qed.

Lemma var_free_in_named_among (x : variable) (d : named) :
  var_free_in_named x d <-> among x (free_variables_of_named d) = true.
Proof.
  destruct d; cbn [var_free_in_named free_variables_of_named];
    first
      [ apply var_free_in_simple_among | apply var_free_in_simples_among
      | apply var_free_in_value_slots_among ].
Qed.

Lemma var_free_in_static_const_among (x : variable) (const : static_const) :
  var_free_in_static_const x const
  <-> among x (free_variables_of_static_const const) = true.
Proof.
  destruct const;
    cbn [var_free_in_static_const free_variables_of_static_const];
    first
      [ apply var_free_in_simple_among | apply var_free_in_simples_among
      | apply var_free_in_value_slots_among
      | simpl; split; [ intros [] | discriminate ] ].
Qed.

Lemma var_free_in_bindings_among (x : variable)
    (bindings : list static_binding) :
  Exists (var_free_in_static_binding x) bindings
  <-> among x (free_variables_of_constants bindings) = true.
Proof.
  unfold free_variables_of_constants.
  rewrite among_flat_map. apply Exists_equiv.
  intros [id c | id | s const]; cbn [var_free_in_static_binding].
  - simpl. split; [ intros [] | discriminate ].
  - simpl. split; [ intros [] | discriminate ].
  - apply var_free_in_static_const_among.
Qed.

Fixpoint var_free_agrees_expr (x : variable) (e : expr) {struct e} :
  code_closed e = true ->
  (var_free_in_expr x e <-> among x (free_variables_of_expr e) = true)
with var_free_agrees_handler (x : variable) (h : continuation_handler)
    {struct h} :
  handler_code_closed h = true ->
  (var_free_in_handler x h
   <-> among x (free_variables_of_handler h) = true).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros Hcode; simpl in Hcode;
      cbn [var_free_in_expr free_variables_of_expr].
    + (* Let *)
      rewrite among_app, orb_true_iff, among_without, andb_true_iff.
      rewrite negb_true_iff.
      pose proof (var_free_in_named_among x d) as Hd.
      pose proof (var_free_agrees_expr x body Hcode) as Hbody.
      pose proof (not_among_singleton x y) as Hy.
      tauto.
    + (* Let_static *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hbindings Hbody].
      destruct (closed_code_free_names bindings Hbindings) as [-> _].
      rewrite among_app, orb_true_iff.
      pose proof (var_free_in_bindings_among x bindings) as Hgroup.
      unfold free_variables_of_constants in Hgroup.
      pose proof (var_free_agrees_expr x body Hbody) as Hbody'.
      tauto.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hh Hbody].
      rewrite among_app, orb_true_iff.
      pose proof (var_free_agrees_handler x h Hh) as Hh'.
      pose proof (var_free_agrees_expr x body Hbody) as Hbody'.
      tauto.
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hhandlers Hbody].
      rewrite among_app, orb_true_iff.
      pose proof (var_free_agrees_expr x body Hbody) as Hbody'.
      assert (Hgroup :
        any_handler (var_free_in_handler x) handlers
        <-> among x
              (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
            = true).
      { clear Hbody Hbody'.
        induction handlers as [ | [k h] handlers IH];
          [ simpl; split; [ intros [] | discriminate ] | ].
        simpl in Hhandlers. apply andb_true_iff in Hhandlers.
        destruct Hhandlers as [Hh Hrest].
        cbn [any_handler flat_map]. rewrite among_app, orb_true_iff.
        pose proof (var_free_agrees_handler x h Hh) as Hh'.
        pose proof (IH Hrest) as Hrest'.
        tauto. }
      tauto.
    + (* Apply *)
      unfold free_variables_of_apply. rewrite !among_app, !orb_true_iff.
      pose proof (var_free_in_simple_among x (ap_callee ap)) as Hcallee.
      pose proof (var_free_in_simples_among x (ap_args ap)) as Hargs.
      pose proof (var_free_in_simples_among x (exn_extra_simples ap))
        as Hextra.
      tauto.
    + (* Apply_cont *)
      unfold free_variables_of_apply_cont. apply var_free_in_simples_among.
    + (* Switch *)
      rewrite among_app, orb_true_iff, among_flat_map.
      pose proof (var_free_in_simple_among x scrutinee) as Hscrutinee.
      assert (Harms :
        Exists (fun arm => var_free_in_simples x (ac_args (snd arm))) arms
        <-> Exists
              (fun arm : Z * apply_cont =>
                 among x (free_variables_of_apply_cont (snd arm)) = true)
              arms).
      { apply Exists_equiv. intros arm. apply var_free_in_simples_among. }
      tauto.
    + (* Invalid *)
      simpl. split; [ intros [] | discriminate ].
  - destruct h as [params metadata handler]. intros Hcode. simpl in Hcode.
    cbn [var_free_in_handler free_variables_of_handler].
    rewrite among_without, andb_true_iff, negb_true_iff.
    pose proof (var_free_agrees_expr x handler Hcode) as Hhandler.
    pose proof (not_among_in x params) as Hparams.
    tauto.
Qed.

Theorem var_free_agrees : var_free_agrees_statement.
Proof. exact var_free_agrees_expr. Qed.

(* ======================================================================== *)
(* The relation on continuations                                            *)
(* ======================================================================== *)

Fixpoint cont_free_agrees_expr (k : continuation) (e : expr) {struct e} :
  code_closed e = true ->
  (cont_free_in_expr k e
   <-> among k (free_continuations_of_expr e) = true)
with cont_free_agrees_handler (k : continuation) (h : continuation_handler)
    {struct h} :
  handler_code_closed h = true ->
  (cont_free_in_handler k h
   <-> among k (free_continuations_of_handler h) = true).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros Hcode; simpl in Hcode;
      cbn [cont_free_in_expr free_continuations_of_expr].
    + (* Let *)
      exact (cont_free_agrees_expr k body Hcode).
    + (* Let_static *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hbindings Hbody].
      destruct (closed_code_free_names bindings Hbindings) as [_ ->].
      exact (cont_free_agrees_expr k body Hbody).
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hh Hbody].
      rewrite among_app, orb_true_iff, among_without, andb_true_iff.
      rewrite negb_true_iff.
      pose proof (cont_free_agrees_handler k h Hh) as Hh'.
      pose proof (cont_free_agrees_expr k body Hbody) as Hbody'.
      pose proof (not_among_singleton k k') as Hk'.
      tauto.
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hhandlers Hbody].
      rewrite among_without, andb_true_iff, negb_true_iff, among_app.
      rewrite orb_true_iff.
      pose proof (cont_free_agrees_expr k body Hbody) as Hbody'.
      assert (Hunbound :
        lookup k handlers = None <-> among k (map fst handlers) = false).
      { rewrite <- is_bound_among. unfold is_bound.
        destruct (lookup k handlers); split; congruence. }
      assert (Hgroup :
        any_handler (cont_free_in_handler k) handlers
        <-> among k
              (flat_map (fun '(_, h) => free_continuations_of_handler h)
                 handlers)
            = true).
      { clear Hbody Hbody' Hunbound.
        induction handlers as [ | [k0 h] handlers IH];
          [ simpl; split; [ intros [] | discriminate ] | ].
        simpl in Hhandlers. apply andb_true_iff in Hhandlers.
        destruct Hhandlers as [Hh Hrest].
        cbn [any_handler flat_map]. rewrite among_app, orb_true_iff.
        pose proof (cont_free_agrees_handler k h Hh) as Hh'.
        pose proof (IH Hrest) as Hrest'.
        tauto. }
      tauto.
    + (* Apply *)
      rewrite among_In. simpl. split.
      * intros [-> | ->]; [ left | right; left ]; reflexivity.
      * intros [<- | [<- | []]]; [ left | right ]; reflexivity.
    + (* Apply_cont *)
      symmetry. apply among_In.
    + (* Switch *)
      rewrite among_In, in_flat_map, Exists_exists.
      split; intros [arm [Hin Hk]]; exists arm; split; assumption.
    + (* Invalid *)
      simpl. split; [ intros [] | discriminate ].
  - destruct h as [params metadata handler]. intros Hcode.
    exact (cont_free_agrees_expr k handler Hcode).
Qed.

Theorem cont_free_agrees : cont_free_agrees_statement.
Proof. exact cont_free_agrees_expr. Qed.

(* ======================================================================== *)
(* Free by a relation is among the free names, whatever the code            *)
(* ======================================================================== *)

Lemma var_free_in_bindings_among_all (x : variable)
    (bindings : list static_binding) :
  Exists (var_free_in_static_binding x) bindings ->
  among x (flat_map free_variables_of_binding bindings) = true.
Proof.
  intros Hfree. apply among_flat_map. apply Exists_exists in Hfree.
  destruct Hfree as [binding [Hin Hbinding]]. apply Exists_exists.
  exists binding. split; [ exact Hin | ].
  destruct binding as [id c | id | s const];
    cbn [var_free_in_static_binding] in Hbinding;
    [ destruct Hbinding | destruct Hbinding | ].
  apply var_free_in_static_const_among, Hbinding.
Qed.

Fixpoint var_free_among_expr (x : variable) (e : expr) {struct e} :
  var_free_in_expr x e -> among x (free_variables_of_expr e) = true
with var_free_among_handler (x : variable) (h : continuation_handler)
    {struct h} :
  var_free_in_handler x h -> among x (free_variables_of_handler h) = true.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [var_free_in_expr free_variables_of_expr].
    + (* Let *)
      rewrite among_app, orb_true_iff, among_without, andb_true_iff.
      rewrite negb_true_iff.
      pose proof (var_free_in_named_among x d) as Hd.
      pose proof (var_free_among_expr x body) as Hbody.
      pose proof (not_among_singleton x y) as Hy.
      tauto.
    + (* Let_static *)
      rewrite among_app, orb_true_iff.
      pose proof (var_free_in_bindings_among_all x bindings) as Hgroup.
      pose proof (var_free_among_expr x body) as Hbody.
      tauto.
    + (* Let_cont_non_recursive *)
      rewrite among_app, orb_true_iff.
      pose proof (var_free_among_handler x h) as Hh.
      pose proof (var_free_among_expr x body) as Hbody.
      tauto.
    + (* Let_cont_recursive *)
      rewrite among_app, orb_true_iff.
      pose proof (var_free_among_expr x body) as Hbody.
      assert (Hgroup :
        any_handler (var_free_in_handler x) handlers ->
        among x
          (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
        = true).
      { clear Hbody.
        induction handlers as [ | [k h] handlers IH]; [ intros [] | ].
        cbn [any_handler flat_map]. rewrite among_app, orb_true_iff.
        pose proof (var_free_among_handler x h) as Hh.
        tauto. }
      tauto.
    + (* Apply *)
      unfold free_variables_of_apply. rewrite !among_app, !orb_true_iff.
      pose proof (var_free_in_simple_among x (ap_callee ap)) as Hcallee.
      pose proof (var_free_in_simples_among x (ap_args ap)) as Hargs.
      pose proof (var_free_in_simples_among x (exn_extra_simples ap))
        as Hextra.
      tauto.
    + (* Apply_cont *)
      unfold free_variables_of_apply_cont. apply var_free_in_simples_among.
    + (* Switch *)
      rewrite among_app, orb_true_iff, among_flat_map.
      pose proof (var_free_in_simple_among x scrutinee) as Hscrutinee.
      assert (Harms :
        Exists (fun arm => var_free_in_simples x (ac_args (snd arm))) arms
        <-> Exists
              (fun arm : Z * apply_cont =>
                 among x (free_variables_of_apply_cont (snd arm)) = true)
              arms).
      { apply Exists_equiv. intros arm. apply var_free_in_simples_among. }
      tauto.
    + (* Invalid *)
      intros [].
  - destruct h as [params metadata handler].
    cbn [var_free_in_handler free_variables_of_handler].
    rewrite among_without, andb_true_iff, negb_true_iff.
    pose proof (var_free_among_expr x handler) as Hhandler.
    pose proof (not_among_in x params) as Hparams.
    tauto.
Qed.

Theorem var_free_among : var_free_among_statement.
Proof. exact var_free_among_expr. Qed.

Fixpoint cont_free_among_expr (k : continuation) (e : expr) {struct e} :
  cont_free_in_expr k e -> among k (free_continuations_of_expr e) = true
with cont_free_among_handler (k : continuation) (h : continuation_handler)
    {struct h} :
  cont_free_in_handler k h ->
  among k (free_continuations_of_handler h) = true.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [cont_free_in_expr free_continuations_of_expr].
    + (* Let *)
      exact (cont_free_among_expr k body).
    + (* Let_static *)
      intros Hbody. rewrite among_app, (cont_free_among_expr k body Hbody).
      apply orb_true_r.
    + (* Let_cont_non_recursive *)
      rewrite among_app, orb_true_iff, among_without, andb_true_iff.
      rewrite negb_true_iff.
      pose proof (cont_free_among_handler k h) as Hh.
      pose proof (cont_free_among_expr k body) as Hbody.
      pose proof (not_among_singleton k k') as Hk'.
      tauto.
    + (* Let_cont_recursive *)
      rewrite among_without, andb_true_iff, negb_true_iff, among_app.
      rewrite orb_true_iff.
      pose proof (cont_free_among_expr k body) as Hbody.
      assert (Hunbound :
        lookup k handlers = None <-> among k (map fst handlers) = false).
      { rewrite <- is_bound_among. unfold is_bound.
        destruct (lookup k handlers); split; congruence. }
      assert (Hgroup :
        any_handler (cont_free_in_handler k) handlers ->
        among k
          (flat_map (fun '(_, h) => free_continuations_of_handler h)
             handlers)
        = true).
      { clear Hbody Hunbound.
        induction handlers as [ | [k0 h] handlers IH]; [ intros [] | ].
        cbn [any_handler flat_map]. rewrite among_app, orb_true_iff.
        pose proof (cont_free_among_handler k h) as Hh.
        tauto. }
      tauto.
    + (* Apply *)
      rewrite among_In. simpl.
      intros [-> | ->]; [ left | right; left ]; reflexivity.
    + (* Apply_cont *)
      apply among_In.
    + (* Switch *)
      rewrite among_In, in_flat_map, Exists_exists.
      intros [arm [Hin Hk]]. exists arm. split; assumption.
    + (* Invalid *)
      intros [].
  - destruct h as [params metadata handler].
    exact (cont_free_among_expr k handler).
Qed.

Theorem cont_free_among : cont_free_among_statement.
Proof. exact cont_free_among_expr. Qed.

(* ======================================================================== *)
(* The free continuations as a list                                         *)
(* ======================================================================== *)

Fixpoint free_continuations_agree_expr (e : expr) {struct e} :
  code_closed e = true ->
  free_continuations_of_expr e = free_continuations e
with free_continuations_agree_handler (h : continuation_handler)
    {struct h} :
  handler_code_closed h = true ->
  free_continuations_of_handler h = handler_free_continuations h.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros Hcode; simpl in Hcode;
      cbn [free_continuations_of_expr free_continuations];
      try reflexivity.
    + (* Let *)
      exact (free_continuations_agree_expr body Hcode).
    + (* Let_static *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hbindings Hbody].
      destruct (closed_code_free_names bindings Hbindings) as [_ ->].
      exact (free_continuations_agree_expr body Hbody).
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hh Hbody].
      rewrite (free_continuations_agree_handler h Hh).
      rewrite (free_continuations_agree_expr body Hbody). reflexivity.
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hhandlers Hbody].
      rewrite (free_continuations_agree_expr body Hbody).
      assert (Hgroup :
        flat_map (fun '(_, h) => free_continuations_of_handler h) handlers
        = flat_map (fun '(_, h) => handler_free_continuations h) handlers).
      { clear Hbody.
        induction handlers as [ | [k0 h] handlers IH]; [ reflexivity | ].
        simpl in Hhandlers. apply andb_true_iff in Hhandlers.
        destruct Hhandlers as [Hh Hrest].
        cbn [flat_map].
        rewrite (free_continuations_agree_handler h Hh), (IH Hrest).
        reflexivity. }
      rewrite Hgroup. reflexivity.
  - destruct h as [params metadata handler]. intros Hcode.
    exact (free_continuations_agree_expr handler Hcode).
Qed.

Theorem free_continuations_agree : free_continuations_agree_statement.
Proof. exact free_continuations_agree_expr. Qed.

(* ======================================================================== *)
(* The check of scope                                                       *)
(* ======================================================================== *)

Lemma simple_in_scope_among (vs : list variable) (s : simple) :
  simple_in_scope vs s = all_among (free_variables_of_simple s) vs.
Proof.
  destruct s as [x | sym | c]; [ | reflexivity | reflexivity ].
  cbn [simple_in_scope free_variables_of_simple].
  rewrite all_among_cons. symmetry. apply andb_true_r.
Qed.

Lemma simples_in_scope_among (vs : list variable) (args : list simple) :
  simples_in_scope vs args = all_among (free_variables_of_simples args) vs.
Proof.
  unfold simples_in_scope, free_variables_of_simples.
  induction args as [ | s args IH]; [ reflexivity | ].
  cbn [forallb flat_map].
  rewrite all_among_app, simple_in_scope_among, IH. reflexivity.
Qed.

Lemma value_slots_in_scope_among (vs : list variable)
    (slots : list (value_slot * simple)) :
  forallb (fun slot => simple_in_scope vs (snd slot)) slots
  = all_among (free_variables_of_value_slots slots) vs.
Proof.
  unfold free_variables_of_value_slots.
  induction slots as [ | slot slots IH]; [ reflexivity | ].
  cbn [forallb flat_map].
  rewrite all_among_app, simple_in_scope_among, IH. reflexivity.
Qed.

Lemma named_in_scope_among (vs : list variable) (d : named) :
  named_in_scope vs d = all_among (free_variables_of_named d) vs.
Proof.
  destruct d; cbn [named_in_scope free_variables_of_named];
    first
      [ apply simple_in_scope_among | apply simples_in_scope_among
      | apply value_slots_in_scope_among ].
Qed.

Lemma static_bindings_in_scope_among (vs : list variable)
    (bindings : list static_binding) :
  forallb (static_binding_in_scope vs) bindings
  = all_among (free_variables_of_constants bindings) vs.
Proof.
  unfold free_variables_of_constants.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [forallb flat_map]. rewrite all_among_app, IH. f_equal.
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    cbn [static_binding_in_scope free_variables_of_static_const];
    first
      [ reflexivity | apply value_slots_in_scope_among
      | apply simples_in_scope_among | apply simple_in_scope_among ].
Qed.

Lemma arms_in_scope_among (vs : list variable) (ks : list continuation)
    (arms : list (Z * apply_cont)) :
  forallb (fun arm => apply_cont_in_scope vs ks (snd arm)) arms = true
  <-> all_among
        (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms)
        vs
      = true
      ∧ all_among
          (flat_map (fun arm => apply_cont_continuations (snd arm)) arms) ks
        = true.
Proof.
  induction arms as [ | [i ac] arms IH]; [ simpl; tauto | ].
  cbn [forallb flat_map snd].
  rewrite andb_true_iff, IH, !all_among_app, !andb_true_iff.
  unfold apply_cont_in_scope at 1.
  rewrite andb_true_iff, simples_in_scope_among.
  unfold free_variables_of_apply_cont. tauto.
Qed.

Fixpoint expr_in_scope_names_among (e : expr) {struct e} :
  ∀ (vs : list variable) (ks : list continuation),
    code_closed e = true ->
    (expr_in_scope vs ks e = true <-> names_among vs ks e = true)
with handler_in_scope_names_among (h : continuation_handler) {struct h} :
  ∀ (vs : list variable) (ks : list continuation),
    handler_code_closed h = true ->
    (handler_in_scope vs ks h = true <-> handler_names_among vs ks h = true).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vs ks Hcode; simpl in Hcode; unfold names_among;
      cbn [expr_in_scope free_variables_of_expr free_continuations_of_expr].
    + (* Let *)
      rewrite !andb_true_iff, all_among_app, andb_true_iff.
      rewrite all_among_without, named_in_scope_among. cbn [app].
      pose proof (expr_in_scope_names_among body (x :: vs) ks Hcode) as Hbody.
      unfold names_among in Hbody. rewrite andb_true_iff in Hbody.
      tauto.
    + (* Let_static *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hbindings Hbody].
      destruct (closed_code_free_names bindings Hbindings) as [-> ->].
      cbn [app]. rewrite !andb_true_iff, all_among_app, andb_true_iff.
      rewrite static_bindings_in_scope_among.
      unfold free_variables_of_constants.
      pose proof (expr_in_scope_names_among body vs ks Hbody) as Hbody'.
      unfold names_among in Hbody'. rewrite andb_true_iff in Hbody'.
      tauto.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hh Hbody].
      rewrite !andb_true_iff, !all_among_app, !andb_true_iff.
      rewrite all_among_without. cbn [app].
      pose proof (handler_in_scope_names_among h vs ks Hh) as Hh'.
      unfold handler_names_among in Hh'. rewrite andb_true_iff in Hh'.
      pose proof (expr_in_scope_names_among body vs (k :: ks) Hbody)
        as Hbody'.
      unfold names_among in Hbody'. rewrite andb_true_iff in Hbody'.
      tauto.
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hcode. destruct Hcode as [Hhandlers Hbody].
      cbv zeta.
      rewrite !andb_true_iff, all_among_without, !all_among_app.
      rewrite !andb_true_iff.
      pose proof
        (expr_in_scope_names_among body vs (map fst handlers ++ ks) Hbody)
        as Hbody'.
      unfold names_among in Hbody'. rewrite andb_true_iff in Hbody'.
      assert (Hgroup :
        forallb
          (fun '(_, h) => handler_in_scope vs (map fst handlers ++ ks) h)
          handlers
        = true
        <-> all_among
              (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
              vs
            = true
            ∧ all_among
                (flat_map (fun '(_, h) => free_continuations_of_handler h)
                   handlers)
                (map fst handlers ++ ks)
              = true).
      { clear Hbody Hbody'. generalize (map fst handlers ++ ks) as ks'.
        intros ks'.
        induction handlers as [ | [k0 h] handlers IH]; [ simpl; tauto | ].
        simpl in Hhandlers. apply andb_true_iff in Hhandlers.
        destruct Hhandlers as [Hh Hrest].
        cbn [forallb flat_map].
        rewrite andb_true_iff, !all_among_app, !andb_true_iff.
        pose proof (handler_in_scope_names_among h vs ks' Hh) as Hh'.
        unfold handler_names_among in Hh'. rewrite andb_true_iff in Hh'.
        pose proof (IH Hrest) as Hrest'.
        tauto. }
      tauto.
    + (* Apply *)
      unfold free_variables_of_apply.
      rewrite !andb_true_iff, !all_among_app, !andb_true_iff.
      rewrite simple_in_scope_among, !simples_in_scope_among.
      rewrite !all_among_cons, !andb_true_iff.
      assert (Hnil : all_among [] ks = true) by reflexivity.
      tauto.
    + (* Apply_cont *)
      unfold apply_cont_in_scope, free_variables_of_apply_cont.
      rewrite !andb_true_iff, simples_in_scope_among.
      tauto.
    + (* Switch *)
      rewrite !andb_true_iff, all_among_app, andb_true_iff.
      rewrite simple_in_scope_among, arms_in_scope_among. tauto.
    + (* Invalid *)
      simpl. tauto.
  - destruct h as [params metadata handler]. intros vs ks Hcode.
    simpl in Hcode.
    rewrite handler_names_among_handler. cbn [handler_in_scope].
    exact (expr_in_scope_names_among handler (params ++ vs) ks Hcode).
Qed.

Theorem expr_in_scope_agrees : expr_in_scope_agrees_statement.
Proof.
  intros vs ks e Hcode. apply eq_iff_eq_true.
  exact (expr_in_scope_names_among e vs ks Hcode).
Qed.
