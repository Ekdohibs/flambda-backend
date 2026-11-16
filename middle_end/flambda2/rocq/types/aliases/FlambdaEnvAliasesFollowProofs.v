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

(* Proofs of the statements of FlambdaEnvAliasesFollowSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaClosuresProofs
  FlambdaContextProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaNormaliseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaEnvAliasesFollowSpec.
Import ListNotations.

(* ======================================================================== *)
(* Binding a variable that is not mentioned                                 *)
(* ======================================================================== *)

Lemma eval_simple_bind (vars : env) (x : variable) (v : value) (s : simple) :
  var_in_simple x s = false ->
  eval_simple ((x, v) :: vars) s = eval_simple vars s.
Proof.
  destruct s as [y | sym | c]; simpl; try reflexivity.
  intros Hx. rewrite Pos.eqb_sym, Hx. reflexivity.
Qed.

Lemma eval_value_slots_bind (vars : env) (x : variable) (v : value)
    (slots : list (value_slot * simple)) :
  var_in_value_slots x slots = false ->
  eval_value_slots ((x, v) :: vars) slots = eval_value_slots vars slots.
Proof.
  unfold var_in_value_slots.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  intros Hx. apply orb_false_iff in Hx. destruct Hx as [Hs Hslots].
  rewrite (eval_simple_bind vars x v s Hs), (IH Hslots). reflexivity.
Qed.

Lemma fold_left_invariant {A B : Type} (P : A -> Prop) (f : A -> B -> A)
    (l : list B) :
  ∀ init : A,
    P init ->
    (∀ (a : A) (b : B), b ∈ l -> P a -> P (f a b)) ->
    P (List.fold_left f l init).
Proof.
  induction l as [ | b l IH]; intros init Hinit Hstep; simpl;
    [ exact Hinit | ].
  apply IH; [ apply Hstep; [ left; reflexivity | exact Hinit ] | ].
  intros a b' Hin. apply Hstep. right. exact Hin.
Qed.

(* ======================================================================== *)
(* The aliases                                                              *)
(* ======================================================================== *)

Lemma empty_unmentioned (n : name) : ~ aliases_mention Aliases.empty n.
Proof.
  intros [[c H] | [[n' H] | [[c H] | [n' H]]]].
  - discriminate H.
  - discriminate H.
  - unfold listed_alias in H. destruct c; discriminate H.
  - unfold listed_alias in H. destruct n; discriminate H.
Qed.

(* The canonical element of a simple, if it is a name, is the simple or
   a name that the structure speaks of. *)
Lemma mentioned_canonical_element_of (t : Aliases.t) (e : simple) (n : name) :
  canonical_element_of t e = simple_of_name n ->
  aliases_mention t n ∨ simple_of_name n = e.
Proof.
  intros Hcanonical.
  destruct (canonical_cases t e)
    as [[_ Heq] | (n' & c & -> & Hdemoted & _ & Heq)].
  - right. rewrite <- Hcanonical. exact Heq.
  - left. right. left. exists n'. rewrite Heq in Hcanonical. subst c.
    exact Hdemoted.
Qed.

Lemma add_equal_respected
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (vars : env) (e e' : simple) :
  aliases_respected t vars ->
  eval_simple vars e = eval_simple vars e' ->
  aliases_respected (Known.add_equal binding_times t e e') vars.
Proof.
  intros Hrespected Hvalues. unfold Known.add_equal.
  destruct (simple_equal _ _); [ exact Hrespected | ].
  destruct (Aliases.add _ _ t _ _) as [[result | ] | ] eqn:Hadd;
    try exact Hrespected.
  apply (add_respected unit Known.binding_time_resolver binding_times t _ _
    result vars Hadd Hrespected).
  rewrite !(respected_canonical_element_of t vars _ Hrespected).
  exact Hvalues.
Qed.

Lemma add_equal_mentioned
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (e e' : simple) (n : name) :
  aliases_mention (Known.add_equal binding_times t e e') n ->
  aliases_mention t n ∨ simple_of_name n = e ∨ simple_of_name n = e'.
Proof.
  unfold Known.add_equal.
  destruct (simple_equal _ _); [ auto | ].
  destruct (Aliases.add _ _ t _ _) as [[result | ] | ] eqn:Hadd; auto.
  intros Hmentioned.
  destruct (add_mentioned unit Known.binding_time_resolver binding_times t _
              _ result n Hadd Hmentioned)
    as [H | [H | H]]; [ auto | | ].
  - destruct (mentioned_canonical_element_of t e n (eq_sym H)); auto.
  - destruct (mentioned_canonical_element_of t e' n (eq_sym H)); auto.
Qed.

Lemma not_the_variable (n : name) (x : variable) :
  negb (name_eqb n (Name_var x)) = true -> Var x <> simple_of_name n.
Proof.
  intros Hn Heq. apply (simple_of_name_inj (Name_var x) n) in Heq. subst n.
  rewrite name_eqb_refl in Hn. discriminate Hn.
Qed.

(* Forgetting a variable: the same environment respects what is left. *)
Lemma forget_in_aliases_respected
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (vars : env) (x : variable) :
  aliases_respected t vars ->
  aliases_respected (Known.forget_in_aliases binding_times t x) vars.
Proof.
  intros Hrespected. unfold Known.forget_in_aliases.
  destruct (negb _); [ exact Hrespected | ]. cbv zeta.
  apply fold_left_invariant; [ apply empty_respected | ].
  intros rebuilt n Hin Hrebuilt.
  assert (Hn :
    eval_simple vars (simple_of_name n)
    = eval_simple vars (Aliases.get_canonical_ignoring_name_mode t n)).
  { symmetry. exact (respected_canonical t vars n Hrespected). }
  destruct (simple_equal (Aliases.get_canonical_ignoring_name_mode t n)
              (Var x)) eqn:Hx.
  - destruct (find _ _) as [n' | ] eqn:Hfind; [ | exact Hrebuilt ].
    apply find_some in Hfind. destruct Hfind as [_ Hx'].
    apply add_equal_respected; [ exact Hrebuilt | ].
    apply simple_equal_eq in Hx. apply simple_equal_eq in Hx'.
    rewrite Hn, Hx, <- Hx'.
    exact (respected_canonical t vars n' Hrespected).
  - apply add_equal_respected; assumption.
Qed.

(* ... and what is left does not speak of the variable. *)
Lemma forget_in_aliases_unmentioned
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (x : variable) :
  ~ aliases_mention (Known.forget_in_aliases binding_times t x) (Name_var x).
Proof.
  unfold Known.forget_in_aliases.
  destruct (among x (aliases_free_variables t)) eqn:Hamong; cbn [negb].
  2: { intros Hmentioned. apply (mentioned_free t x) in Hmentioned.
       congruence. }
  cbv zeta. apply fold_left_invariant; [ apply empty_unmentioned | ].
  intros rebuilt n Hin Hrebuilt.
  apply filter_In in Hin. destruct Hin as [_ Hn].
  destruct (simple_equal (Aliases.get_canonical_ignoring_name_mode t n)
              (Var x)) eqn:Hx.
  - destruct (find _ _) as [n' | ] eqn:Hfind; [ | exact Hrebuilt ].
    apply find_some in Hfind. destruct Hfind as [Hin' _].
    apply filter_In in Hin'. destruct Hin' as [_ Hn'].
    intros Hmentioned. apply add_equal_mentioned in Hmentioned.
    destruct Hmentioned as [H | [H | H]]; [ exact (Hrebuilt H) | | ].
    + exact (not_the_variable n x Hn H).
    + exact (not_the_variable n' x Hn' H).
  - intros Hmentioned. apply add_equal_mentioned in Hmentioned.
    destruct Hmentioned as [H | [H | H]]; [ exact (Hrebuilt H) | | ].
    + exact (not_the_variable n x Hn H).
    + simpl in H. rewrite <- H in Hx. simpl in Hx.
      rewrite Pos.eqb_refl in Hx. discriminate Hx.
Qed.

(* ======================================================================== *)
(* What is known, under binders                                             *)
(* ======================================================================== *)

Theorem empty_known_respected : empty_known_respected_statement.
Proof.
  intros statics vars.
  split; [ apply empty_respected | split; [ intros x c [] | intros s c [] ] ].
Qed.

Theorem known_respected_extend : known_respected_extend_statement.
Proof.
  intros known statics statics' vars (Haliases & Hclosures & Hstatics)
    [_ Hvalues].
  split; [ exact Haliases | split; [ exact Hclosures | ] ].
  intros s c Hin.
  destruct (Hstatics s c Hin) as [Hcode (slot_values & Hslots & Hbound)].
  split; [ exact Hcode | ]. exists slot_values.
  split; [ exact Hslots | exact (Hvalues s _ Hbound) ].
Qed.

Lemma canonical_simple_sound (known : Known.t) {statics : static_env}
    (vars : env) (s : simple) :
  known_respected known statics vars ->
  eval_simple vars (Known.canonical_simple known s) = eval_simple vars s.
Proof.
  intros [Haliases _].
  exact (respected_canonical_element_of _ vars s Haliases).
Qed.

Theorem canonical_simple_respected : canonical_simple_respected_statement.
Proof. intros known statics vars s. apply canonical_simple_sound. Qed.

Lemma closure_respected_bind (vars : env) (x : variable) (v : value)
    (y : variable) (c : Known.closure) :
  closure_respected vars y c ->
  Pos.eqb x y = false ->
  var_in_value_slots x (Known.closure_value_slots c) = false ->
  closure_respected ((x, v) :: vars) y c.
Proof.
  intros [Hcode (slot_values & Hslots & Hlookup)] Hxy Hfree.
  split; [ exact Hcode | ]. exists slot_values. split.
  - rewrite eval_value_slots_bind; assumption.
  - simpl. rewrite Pos.eqb_sym, Hxy. exact Hlookup.
Qed.

Lemma static_closure_respected_bind (statics : static_env) (vars : env)
    (x : variable) (v : value) (s : symbol) (c : Known.static_closure) :
  static_closure_respected statics vars s c ->
  var_in_value_slots x (Known.closure_value_slots (Known.as_closure c))
  = false ->
  static_closure_respected statics ((x, v) :: vars) s c.
Proof.
  intros [Hcode (slot_values & Hslots & Hbound)] Hfree.
  split; [ exact Hcode | ]. exists slot_values.
  split; [ rewrite eval_value_slots_bind; assumption | exact Hbound ].
Qed.

Lemma in_static_closures_forget (known : Known.t) (x : variable)
    (s : symbol) (c : Known.static_closure) :
  (s, c) ∈ Known.static_closures (Known.forget known x) ->
  (s, c) ∈ Known.static_closures known
  ∧ var_in_value_slots x (Known.closure_value_slots (Known.as_closure c))
    = false.
Proof.
  unfold Known.forget. cbn [Known.static_closures]. intros Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. simpl in Hkept.
  apply negb_true_iff in Hkept. auto.
Qed.

Lemma in_closures_forget (known : Known.t) (x y : variable)
    (c : Known.closure) :
  (y, c) ∈ Known.closures (Known.forget known x) ->
  (y, c) ∈ Known.closures known
  ∧ Pos.eqb x y = false
  ∧ var_in_value_slots x (Known.closure_value_slots c) = false.
Proof.
  unfold Known.forget. cbn [Known.closures]. intros Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. simpl in Hkept.
  apply andb_true_iff in Hkept. destruct Hkept as [Hxy Hfree].
  apply negb_true_iff in Hxy. apply negb_true_iff in Hfree. auto.
Qed.

Theorem forget_respected : forget_respected_statement.
Proof.
  intros known statics vars x v (Haliases & Hclosures & Hstatics).
  split; [ | split ].
  - unfold Known.forget. cbn [Known.aliases].
    apply (respected_coincide
      (Known.forget_in_aliases (Known.binding_times known)
         (Known.aliases known) x)
      vars ((x, v) :: vars));
      [ | apply forget_in_aliases_respected, Haliases ].
    intros n Hmentioned. destruct n as [y | sym]; [ | reflexivity ].
    unfold value_of_name. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ | reflexivity ].
    exfalso. exact (forget_in_aliases_unmentioned _ _ _ Hmentioned).
  - intros y c Hin.
    destruct (in_closures_forget known x y c Hin) as (Hin' & Hxy & Hfree).
    apply closure_respected_bind;
      [ exact (Hclosures y c Hin') | exact Hxy | exact Hfree ].
  - intros s c Hin.
    destruct (in_static_closures_forget known x s c Hin) as [Hin' Hfree].
    apply static_closure_respected_bind;
      [ exact (Hstatics s c Hin') | exact Hfree ].
Qed.

Theorem bind_respected : bind_respected_statement.
Proof.
  intros known statics vars x v Hrespected.
  exact (forget_respected known statics vars x v Hrespected).
Qed.

Lemma bind_all_respected (params : list variable) :
  ∀ (known : Known.t) (statics : static_env) (vs : list value)
    (vars handler_vars : env),
    known_respected known statics vars ->
    bind_params params vs vars = Some handler_vars ->
    known_respected (Known.bind_all known params) statics handler_vars.
Proof.
  induction params as [ | param params IH];
    intros known statics [ | v vs] vars handler_vars Hrespected; simpl;
    try discriminate.
  - intros [= <-]. exact Hrespected.
  - apply (IH (Known.bind known param)). apply bind_respected, Hrespected.
Qed.

Theorem under_let_respected : under_let_respected_statement.
Proof.
  intros known statics vars x d v Hrespected Hd.
  pose proof (bind_respected known statics vars x v Hrespected) as Hbound.
  unfold Known.under_let. cbv zeta.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hbound.
  - (* Simple *)
    destruct (var_in_simple x s) eqn:Hfree; [ exact Hbound | ].
    destruct Hbound as (Haliases & Hclosures & Hstatics).
    split; [ | split; [ exact Hclosures | exact Hstatics ] ].
    cbn [Known.aliases Known.with_aliases].
    apply add_equal_respected; [ exact Haliases | ].
    simpl. rewrite Pos.eqb_refl, (eval_simple_bind vars x v s Hfree).
    symmetry. exact Hd.
  - (* Set_of_closures *)
    destruct (var_in_value_slots x slots) eqn:Hfree; [ exact Hbound | ].
    destruct Hbound as (Haliases & Hclosures & Hstatics).
    split; [ exact Haliases | split; [ | exact Hstatics ] ].
    intros y c [[= <- <-] | Hin]; [ | exact (Hclosures y c Hin) ].
    simpl in Hd.
    destruct (eval_value_slots vars slots) as [slot_values | ] eqn:Hslots;
      [ | discriminate Hd ].
    simpl in Hd. injection Hd as <-.
    split; [ discriminate | ]. exists slot_values. split.
    + simpl. rewrite eval_value_slots_bind; assumption.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
  - (* Closure_of_set *)
    destruct (var_in_value_slots x slots) eqn:Hfree; [ exact Hbound | ].
    destruct Hbound as (Haliases & Hclosures & Hstatics).
    split; [ exact Haliases | split; [ | exact Hstatics ] ].
    intros y c [[= <- <-] | Hin]; [ | exact (Hclosures y c Hin) ].
    simpl in Hd.
    destruct (lookup fslot decls) as [code | ] eqn:Hcode;
      [ | discriminate Hd ].
    simpl in Hd.
    destruct (eval_value_slots vars slots) as [slot_values | ] eqn:Hslots;
      [ | discriminate Hd ].
    simpl in Hd. injection Hd as <-.
    split; [ simpl; rewrite Hcode; discriminate | ].
    exists slot_values. split.
    + simpl. rewrite eval_value_slots_bind; assumption.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
  - (* Project_function_slot *)
    destruct closure as [y | sym | k]; try exact Hbound.
    destruct (lookup y (Known.closures (Known.bind known x)))
      as [[code slots | slot decls slots] | ] eqn:Hfind;
      try exact Hbound.
    destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | exact Hbound ].
    pose proof (lookup_in y _ _ Hfind) as Hin.
    destruct (in_closures_forget known x y _ Hin) as (_ & Hxy & Hfree).
    destruct Hbound as (Haliases & Hclosures & Hstatics).
    split; [ exact Haliases | split; [ | exact Hstatics ] ].
    intros z c [[= <- <-] | Hin']; [ | exact (Hclosures z c Hin') ].
    destruct (Hclosures y _ Hin) as [Hcode (slot_values & Hslots & Hlookup)].
    simpl in Hlookup. rewrite Pos.eqb_sym, Hxy in Hlookup.
    simpl in Hd. rewrite Hlookup in Hd. simpl in Hd.
    rewrite Pos.eqb_refl in Hd. simpl in Hcode.
    destruct (lookup move_from decls) as [code | ]; [ | contradiction ].
    simpl in Hd.
    destruct (lookup move_to decls) as [code' | ] eqn:Hcode';
      [ | discriminate Hd ].
    simpl in Hd. injection Hd as <-.
    split; [ simpl; rewrite Hcode'; discriminate | ].
    exists slot_values. split; [ exact Hslots | ].
    simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Theorem under_let_static_respected : under_let_static_respected_statement.
Proof.
  intros known statics extended vars bindings Hrespected Heval.
  destruct
    (known_respected_extend known statics extended vars Hrespected
       (eval_static_bindings_extend vars bindings statics extended Heval))
    as (Haliases & Hclosures & Hstatics).
  split; [ exact Haliases | split; [ exact Hclosures | ] ].
  intros s c Hin. unfold Known.under_let_static in Hin.
  cbn [Known.static_closures] in Hin. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin]; [ | exact (Hstatics s c Hin) ].
  apply in_flat_map in Hin. destruct Hin as (binding & Hbinding & Hentry).
  destruct binding
    as [id code | id
       | sym [code slots | fslot syms decls slots | tag fields | contents
             | bytes | elements]];
    simpl in Hentry; try contradiction;
    destruct Hentry as [[= <- <-] | []];
    destruct
      (eval_static_bindings_binds vars bindings statics extended sym _ Heval
         Hbinding)
      as (allocated & Hconst & Hbound);
    simpl in Hconst.
  - destruct (eval_value_slots vars slots) as [slot_values | ] eqn:Hslots;
      [ | discriminate Hconst ].
    simpl in Hconst. injection Hconst as <-.
    split; [ discriminate | ]. exists slot_values.
    split; [ exact Hslots | exact Hbound ].
  - destruct (lookup fslot decls) as [code | ] eqn:Hcode;
      [ | discriminate Hconst ].
    simpl in Hconst.
    destruct (eval_value_slots vars slots) as [slot_values | ] eqn:Hslots;
      [ | discriminate Hconst ].
    simpl in Hconst. injection Hconst as <-.
    split; [ simpl; rewrite Hcode; discriminate | ]. exists slot_values.
    split; [ exact Hslots | exact Hbound ].
Qed.

(* ======================================================================== *)
(* Following aliases at the head of a term                                  *)
(* ======================================================================== *)

Lemma canonical_simples_sound (known : Known.t) {statics : static_env}
    (vars : env) (args : list simple) :
  known_respected known statics vars ->
  eval_simples vars (map (Known.canonical_simple known) args)
  = eval_simples vars args.
Proof.
  intros Hrespected. induction args as [ | s args IH]; simpl;
    [ reflexivity | ].
  rewrite (canonical_simple_sound known vars s Hrespected), IH. reflexivity.
Qed.

Lemma canonical_value_slots_sound (known : Known.t) {statics : static_env}
    (vars : env) (slots : list (value_slot * simple)) :
  known_respected known statics vars ->
  eval_value_slots vars (canonical_value_slots known slots)
  = eval_value_slots vars slots.
Proof.
  intros Hrespected. induction slots as [ | [slot s] slots IH]; simpl;
    [ reflexivity | ].
  rewrite (canonical_simple_sound known vars s Hrespected), IH. reflexivity.
Qed.

(* What a known closure is: the value of the simple is a closure, made
   by a [Let] or bound to a symbol, of that code and at that slot, with
   the values of the value slots. *)
Lemma find_closure_sound (known : Known.t) (statics : static_env)
    (vars : env) (s : simple) (c : Known.closure) :
  known_respected known statics vars ->
  Known.find_closure known s = Some c ->
  Known.closure_code c <> None
  ∧ ∃ (v : value) (slot_values : list (value_slot * value)),
      eval_value_slots vars (Known.closure_value_slots c) = Some slot_values
      ∧ eval_simple vars s = Some v
      ∧ closure_of statics v
        = option_map (fun code => (code, slot_values)) (Known.closure_code c)
      ∧ ∀ (project_from : function_slot) (slot : value_slot),
          project_value_slot statics project_from slot v
          = if Pos.eqb (Known.closure_slot c) project_from
            then lookup slot slot_values
            else None.
Proof.
  intros (_ & Hclosures & Hstatics) Hfind.
  destruct s as [y | sym | k]; simpl in Hfind; [ | | discriminate Hfind ].
  - destruct (Hclosures y c (lookup_in y c _ Hfind))
      as [Hcode (slot_values & Hslots & Hbound)].
    split; [ exact Hcode | ].
    exists (closure_value c slot_values), slot_values.
    split; [ exact Hslots | split; [ exact Hbound | ] ].
    destruct c as [code slots | fslot decls slots]; simpl;
      [ split; reflexivity | ].
    unfold project_value_slot. simpl. simpl in Hcode.
    destruct (lookup fslot decls); [ split; reflexivity | contradiction ].
  - destruct (lookup sym (Known.static_closures known)) as [sc | ] eqn:Hsym;
      [ | discriminate Hfind ].
    simpl in Hfind. injection Hfind as <-.
    destruct (Hstatics sym sc (lookup_in sym sc _ Hsym))
      as [Hcode (slot_values & Hslots & Hbound)].
    split; [ exact Hcode | ]. exists (Value_symbol sym), slot_values.
    split; [ exact Hslots | split; [ reflexivity | ] ].
    unfold project_value_slot, closure_in_set, closure_of.
    rewrite Hbound. unfold static_closure_value.
    destruct (Known.as_closure sc) as [code slots | fslot decls slots]; simpl;
      [ split; reflexivity | ].
    simpl in Hcode.
    destruct (lookup fslot decls); [ split; reflexivity | contradiction ].
Qed.

(* A slot of evaluated value slots has the value of what it held. *)
Lemma eval_value_slots_lookup (vars : env) (slot : value_slot)
    (slots : list (value_slot * simple)) :
  ∀ (slot_values : list (value_slot * value)) (contents : simple),
    eval_value_slots vars slots = Some slot_values ->
    lookup slot slots = Some contents ->
    lookup slot slot_values = eval_simple vars contents.
Proof.
  induction slots as [ | [slot' s] slots IH]; intros slot_values contents;
    simpl; [ discriminate | ].
  destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
  destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
  simpl. intros [= <-]. simpl.
  destruct (Pos.eqb slot slot').
  - intros [= <-]. symmetry. exact Hs.
  - apply (IH rest contents eq_refl).
Qed.

Lemma canonical_project_sound (known : Known.t) (vars : env)
    (statics : static_env) (slot : value_slot) (closure : simple) :
  known_respected known statics vars ->
  eval_named statics vars (canonical_project known slot closure)
  = eval_named statics vars (Project_value_slot slot closure).
Proof.
  intros Hrespected. unfold canonical_project.
  destruct (Known.find_closure known closure) as [c | ] eqn:Hclosure;
    [ | reflexivity ].
  destruct (lookup slot (Known.closure_value_slots c)) as [contents | ]
    eqn:Hslot; [ | reflexivity ].
  destruct
    (find_closure_sound known statics vars closure c Hrespected Hclosure)
    as [Hcode (v & slot_values & Hslots & Heval & Hclosure_of & _)].
  simpl. rewrite Heval. simpl. rewrite Hclosure_of.
  destruct (Known.closure_code c) as [code | ]; [ | contradiction ].
  simpl. symmetry.
  exact (eval_value_slots_lookup vars slot _ slot_values contents Hslots
    Hslot).
Qed.

Lemma canonical_project_from_sound (known : Known.t) (vars : env)
    (statics : static_env) (project_from : function_slot) (slot : value_slot)
    (closure : simple) :
  known_respected known statics vars ->
  eval_named statics vars
    (canonical_project_from known project_from slot closure)
  = eval_named statics vars
      (Project_value_slot_from project_from slot closure).
Proof.
  intros Hrespected. unfold canonical_project_from.
  destruct (Known.find_closure known closure) as [c | ] eqn:Hclosure;
    [ | reflexivity ].
  destruct (Pos.eqb (Known.closure_slot c) project_from) eqn:Hat;
    [ | reflexivity ].
  destruct (lookup slot (Known.closure_value_slots c)) as [contents | ]
    eqn:Hslot; [ | reflexivity ].
  destruct
    (find_closure_sound known statics vars closure c Hrespected Hclosure)
    as [Hcode (v & slot_values & Hslots & Heval & _ & Hproject)].
  simpl. rewrite Heval. simpl. rewrite Hproject, Hat.
  symmetry.
  exact (eval_value_slots_lookup vars slot _ slot_values contents Hslots
    Hslot).
Qed.

Lemma canonical_project_function_slot_sound (known : Known.t) (vars : env)
    (statics : static_env) (move_from move_to : function_slot)
    (closure : simple) :
  known_respected known statics vars ->
  eval_named statics vars
    (canonical_project_function_slot known move_from move_to closure)
  = eval_named statics vars (Project_function_slot move_from move_to closure).
Proof.
  intros (_ & _ & Hstatics). unfold canonical_project_function_slot.
  destruct closure as [x | s | k]; try reflexivity.
  destruct (lookup s (Known.static_closures known)) as [sc | ] eqn:Hsym;
    [ | reflexivity ].
  destruct (Known.as_closure sc) as [code slots | slot decls slots] eqn:Hsc;
    [ reflexivity | ].
  destruct (Pos.eqb slot move_from && is_bound move_to decls) eqn:Hat;
    [ | reflexivity ].
  destruct (lookup move_to (Known.closure_symbols sc)) as [s' | ] eqn:Hto;
    [ | reflexivity ].
  apply andb_true_iff in Hat. destruct Hat as [Hfrom Hbound_to].
  destruct (Hstatics s sc (lookup_in s sc _ Hsym))
    as [Hcode (slot_values & _ & Hbound)].
  rewrite Hsc in Hcode. simpl in Hcode.
  unfold is_bound in Hbound_to. simpl.
  unfold static_closure_value in Hbound. rewrite Hsc in Hbound.
  rewrite Hbound, Hfrom.
  destruct (lookup slot decls); [ | contradiction ].
  destruct (lookup move_to decls); [ | discriminate Hbound_to ].
  simpl. rewrite Hto. reflexivity.
Qed.

Lemma canonical_named_sound (known : Known.t) (vars : env)
    (statics : static_env) (d : named) :
  known_respected known statics vars ->
  eval_named statics vars (canonical_named known d)
  = eval_named statics vars d.
Proof.
  intros Hrespected.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - exact (canonical_simple_sound known vars s Hrespected).
  - rewrite (canonical_simples_sound known vars args Hrespected).
    reflexivity.
  - rewrite (canonical_value_slots_sound known vars slots Hrespected).
    reflexivity.
  - rewrite (canonical_value_slots_sound known vars slots Hrespected).
    reflexivity.
  - rewrite (canonical_project_sound known vars statics slot _ Hrespected).
    simpl.
    rewrite (canonical_simple_sound known vars closure Hrespected).
    reflexivity.
  - rewrite (canonical_project_from_sound known vars statics pfrom slot _
      Hrespected).
    simpl.
    rewrite (canonical_simple_sound known vars closure Hrespected).
    reflexivity.
  - rewrite (canonical_project_function_slot_sound known vars statics
      move_from move_to _ Hrespected).
    simpl.
    rewrite (canonical_simple_sound known vars closure Hrespected).
    reflexivity.
Qed.

(* The same of the run of the named expression: a primitive is run on the
   values of its arguments, and what is not a primitive has the value that
   [eval_named] gives it, with the heap and the record as they came. *)
Lemma canonical_named_sound_exec (known : Known.t) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (d : named) :
  known_respected known statics vars ->
  exec_named statics hp rd vars (canonical_named known d)
  = exec_named statics hp rd vars d.
Proof.
  intros Hrespected.
  assert (Hnot : ∀ d' : named,
    (∀ (p : prim) (args : list simple), d' <> Prim p args) ->
    (∀ (p : prim) (args : list simple),
      canonical_named known d' <> Prim p args) ->
    exec_named statics hp rd vars (canonical_named known d')
    = exec_named statics hp rd vars d').
  { intros d' Hd' Hcanonical.
    rewrite (exec_named_not_prim statics hp rd vars _ Hcanonical),
      (exec_named_not_prim statics hp rd vars d' Hd'),
      (canonical_named_sound known vars statics d' Hrespected).
    reflexivity. }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    [ | simpl; apply exec_named_args;
        exact (canonical_simples_sound known vars args Hrespected)
      | | | | | ];
    (apply Hnot; [ intros ? ?; discriminate | ]);
    simpl;
    unfold canonical_project, canonical_project_from,
      canonical_project_function_slot;
    intros p args;
    repeat match goal with
           | |- context [match ?scrutinee with _ => _ end] =>
             destruct scrutinee
           end;
    discriminate.
Qed.

(* A call and the same call with its aliases followed and, if the callee
   is a known closure, made direct: they take the same step. *)
Lemma canonical_apply_equivalent (known : Known.t) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  known_respected known statics vars ->
  equivalent (State (Apply ap) vars conts statics hp rd)
    (State (Apply (canonical_apply known ap)) vars conts statics hp rd).
Proof.
  intros Hrespected. apply step_eq_equivalent. rewrite !step_apply.
  assert (Hextras :
    exn_extra_simples (canonical_apply known ap)
    = map (Known.canonical_simple known) (exn_extra_simples ap))
    by (unfold exn_extra_simples; simpl; rewrite !map_map; reflexivity).
  rewrite Hextras. simpl.
  rewrite (canonical_simples_sound known vars (ap_args ap) Hrespected),
    (canonical_simples_sound known vars (exn_extra_simples ap) Hrespected).
  rewrite (canonical_simple_sound known vars (ap_callee ap) Hrespected).
  unfold canonical_call_kind.
  destruct (ap_call_kind ap) as [code | ]; [ reflexivity | ].
  destruct (Known.find_closure known
              (Known.canonical_simple known (ap_callee ap)))
    as [c | ] eqn:Hclosure; [ | reflexivity ].
  destruct (Known.closure_code c) as [code | ] eqn:Hcode; [ | reflexivity ].
  destruct (find_closure_sound known statics vars _ c Hrespected Hclosure)
    as [_ (v & slot_values & _ & Heval & Hclosure_of & _)].
  rewrite (canonical_simple_sound known vars (ap_callee ap) Hrespected)
    in Heval.
  rewrite Heval.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  assert (Hclosure_of' : closure_of statics v = Some (code, slot_values))
    by (rewrite Hclosure_of, Hcode; reflexivity).
  rewrite !(shape_of_call_closure statics _ _ _ _ _
              (closure_not_partial statics _ _ Hclosure_of')),
    (shape_of_closure_call_kind statics _ (Direct code) _ _ _ code
       slot_values Hclosure_of' (or_intror eq_refl)).
  reflexivity.
Qed.

Lemma canonical_static_bindings_sound (known : Known.t)
    {statics0 : static_env} (vars : env) (bindings : list static_binding) :
  known_respected known statics0 vars ->
  ∀ statics : static_env,
    eval_static_bindings vars (map (canonical_static_binding known) bindings)
      statics
    = eval_static_bindings vars bindings statics.
Proof.
  intros Hrespected.
  induction bindings as [ | binding bindings IH]; intros statics;
    [ reflexivity | ].
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    simpl; try apply IH.
  - destruct (lookup id (static_code statics)); [ reflexivity | apply IH ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_value_slots_sound known vars slots Hrespected).
    destruct (eval_value_slots vars slots); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_value_slots_sound known vars slots Hrespected).
    destruct (lookup fslot decls); simpl; [ | reflexivity ].
    destruct (eval_value_slots vars slots); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_simples_sound known vars fields Hrespected).
    destruct (eval_simples vars fields); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_simple_sound known vars s Hrespected).
    destruct (eval_simple vars s); simpl; [ apply IH | reflexivity ].
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    apply IH.
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite (canonical_simples_sound known vars elements Hrespected).
    destruct (eval_simples vars elements); simpl; [ apply IH | reflexivity ].
Qed.

Lemma find_arm_canonical (known : Known.t) (i : Z)
    (arms : list (Z * apply_cont)) :
  find_arm i
    (map (fun arm => (fst arm, canonical_apply_cont known (snd arm))) arms)
  = option_map (canonical_apply_cont known) (find_arm i arms).
Proof.
  induction arms as [ | [discriminant ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i discriminant); [ reflexivity | exact IH ].
Qed.

Lemma canonical_apply_cont_equivalent (known : Known.t) (ac : apply_cont)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  known_respected known statics vars ->
  equivalent (State (Apply_cont ac) vars conts statics hp rd)
    (State (Apply_cont (canonical_apply_cont known ac)) vars conts statics hp
       rd).
Proof.
  intros Hrespected. apply step_eq_equivalent. rewrite !step_apply_cont.
  simpl.
  rewrite (canonical_simples_sound known vars (ac_args ac) Hrespected).
  reflexivity.
Qed.

(* ======================================================================== *)
(* Under binders                                                            *)
(* ======================================================================== *)

(* Equivalence in the environments that agree with what is known. *)
Definition equivalent_where (known : Known.t) (e e' : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    known_respected known statics vars ->
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

(* [under_let_respected] of the run of the named expression: nothing is
   recorded of a primitive, and what is not a primitive has the value that
   [eval_named] gives it. *)
Lemma under_let_respected_exec (known : Known.t) (statics : static_env)
    (vars : env) (x : variable) (d : named) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  known_respected known statics vars ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  known_respected (Known.under_let known x d) statics ((x, v) :: vars).
Proof.
  intros Hrespected Hd.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (rewrite exec_named_not_prim in Hd by (intros ? ?; discriminate);
         destruct (eval_named statics vars _) as [w | ] eqn:Hw;
           [ | discriminate ];
         injection Hd as <- <- <-;
         exact (under_let_respected known statics vars x _ w Hrespected Hw)).
  exact (bind_respected known statics vars x v Hrespected).
Qed.

Lemma let_equivalent_where (known : Known.t) (x : variable)
    (kind : flambda_kind) (d : named) (body body' : expr) :
  equivalent_where (Known.under_let known x d) body body' ->
  equivalent_where known (Let x kind d body) (Let x kind d body').
Proof.
  intros Hbody vars conts statics hp rd Hrespected.
  apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' Hd _ ].
  apply Hbody.
  exact (under_let_respected_exec known statics vars x d hp hp' rd rd' v
    Hrespected Hd).
Qed.

Lemma let_static_equivalent_where (known : Known.t)
    (bindings : list static_binding) (body body' : expr) :
  equivalent_where (Known.under_let_static known bindings) body body' ->
  equivalent_where known (Let_static bindings body)
    (Let_static bindings body').
Proof.
  intros Hbody vars conts statics hp rd Hrespected.
  apply let_static_equivalent_step.
  destruct (eval_static_bindings vars bindings statics) as [extended | ]
    eqn:Heval; [ | exact I ].
  apply Hbody.
  exact (under_let_static_respected known statics extended vars bindings
    Hrespected Heval).
Qed.

(* A handler may be replaced by one equivalent to it under what is known
   where it is defined, with its parameters bound. *)
Lemma let_cont_handler_equivalent_where (known : Known.t) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) :
  equivalent_where (Known.bind_all known params) handler handler' ->
  equivalent_where known
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hhandler vars conts statics hp rd Hrespected.
  apply refines_antisymmetric;
    apply
      (let_cont_handler_refines_where
         (fun statics' => known_respected known statics' vars)
         (fun statics' statics'' => known_respected_extend known _ _ vars));
    try exact Hrespected;
    intros vs handler_vars conts' statics' hp' rd' Hok _ Hbind;
    apply equivalent_refines;
    [ | apply equivalent_sym ];
    apply Hhandler,
      (bind_all_respected params known statics' vs vars handler_vars Hok
         Hbind).
Qed.

Lemma equivalent_where_trans (known : Known.t) (e e' e'' : expr) :
  equivalent_where known e e' -> equivalent_where known e' e'' ->
  equivalent_where known e e''.
Proof.
  intros H H' vars conts statics hp rd Hrespected.
  exact (equivalent_trans _ _ _ (H vars conts statics hp rd_ Hrespected)
    (H' vars conts statics hp rd_ Hrespected)).
Qed.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

(* At each node: the head with its aliases followed is equivalent, by
   evaluating the same; then the parts under binders, by the induction. *)
Fixpoint follow_aliases_equivalent_where (e : expr) :
  ∀ known : Known.t, equivalent_where known e (follow_aliases known e)
with follow_aliases_in_handler_equivalent_where (h : continuation_handler) :
  ∀ (known : Known.t) (k : continuation) (body : expr),
    equivalent_where known (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (follow_aliases_in_handler known h) body).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; simpl.
    + (* Let *)
      apply (equivalent_where_trans known _
        (Let x kind (canonical_named known d) body)).
      * intros vars conts statics hp rd Hrespected.
        apply let_equivalent_step_exec;
          [ apply (canonical_named_sound_exec known vars statics hp rd d
              Hrespected)
          | intros v hp' rd' _ _; apply equivalent_refl ].
      * apply let_equivalent_where, follow_aliases_equivalent_where.
    + (* Let_static *)
      apply (equivalent_where_trans known _
        (Let_static (map (canonical_static_binding known) bindings) body)).
      * intros vars conts statics hp rd Hrespected. apply step_eq_equivalent.
        rewrite !step_let_static,
          (canonical_static_bindings_sound known vars bindings Hrespected).
        reflexivity.
      * apply let_static_equivalent_where, follow_aliases_equivalent_where.
    + (* Let_cont_non_recursive *)
      apply (equivalent_where_trans known _
        (Let_cont_non_recursive k (follow_aliases_in_handler known h) body));
        [ apply follow_aliases_in_handler_equivalent_where | ].
      intros vars conts statics hp rd Hrespected.
      apply let_cont_non_recursive_equivalent_step.
      apply (follow_aliases_equivalent_where body known), Hrespected.
    + (* Let_cont_recursive *)
      intros vars conts statics hp rd Hrespected.
      apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply (follow_aliases_equivalent_where body known), Hrespected.
    + (* Apply *)
      intros vars conts statics hp rd Hrespected.
      apply canonical_apply_equivalent, Hrespected.
    + (* Apply_cont *)
      intros vars conts statics hp rd Hrespected.
      apply canonical_apply_cont_equivalent, Hrespected.
    + (* Switch *)
      intros vars conts statics hp rd Hrespected.
      destruct (eval_simple vars scrutinee) as [v | ] eqn:Hscrutinee;
        [ destruct (naked_immediate_of_value v) as [i | ] eqn:Himmediate;
          [ destruct (find_arm i arms) as [ac | ] eqn:Hfind | ]
        | ].
      * eapply equivalent_step;
          [ rewrite step_switch, Hscrutinee, Himmediate, Hfind; reflexivity
          | rewrite step_switch,
              (canonical_simple_sound known vars scrutinee Hrespected),
              Hscrutinee, Himmediate, find_arm_canonical, Hfind;
            reflexivity
          | ].
        apply canonical_apply_cont_equivalent, Hrespected.
      * apply step_eq_equivalent.
        rewrite !step_switch,
          (canonical_simple_sound known vars scrutinee Hrespected),
          Hscrutinee, Himmediate, find_arm_canonical, Hfind.
        reflexivity.
      * apply step_eq_equivalent.
        rewrite !step_switch,
          (canonical_simple_sound known vars scrutinee Hrespected),
          Hscrutinee, Himmediate.
        reflexivity.
      * apply step_eq_equivalent.
        rewrite !step_switch,
          (canonical_simple_sound known vars scrutinee Hrespected),
          Hscrutinee.
        reflexivity.
    + (* Invalid *)
      intros vars conts statics hp rd Hrespected. apply equivalent_refl.
  - destruct h as [params metadata handler]. intros known k body. simpl.
    apply let_cont_handler_equivalent_where, follow_aliases_equivalent_where.
Qed.

Theorem follow_aliases_respected : follow_aliases_respected_statement.
Proof.
  intros known e vars conts statics hp rd Hrespected.
  exact (follow_aliases_equivalent_where e known vars conts statics
    hp rd_ Hrespected).
Qed.

Theorem follow_aliases_equivalent : follow_aliases_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  apply follow_aliases_respected, empty_known_respected.
Qed.

(* ======================================================================== *)
(* Consistency, and what forgetting a variable keeps                        *)
(* ======================================================================== *)

Lemma empty_consistent : aliases_consistent Aliases.empty.
Proof.
  exact (proj1 (empty_well_formed unit Known.binding_time_resolver [])).
Qed.

Lemma add_equal_consistent
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (e e' : simple) :
  aliases_consistent t ->
  aliases_consistent (Known.add_equal binding_times t e e').
Proof.
  intros Hconsistent. unfold Known.add_equal.
  destruct (simple_equal _ _); [ exact Hconsistent | ].
  destruct (Aliases.add _ _ t _ _) as [[result | ] | ] eqn:Hadd;
    try exact Hconsistent.
  exact (add_consistent unit Known.binding_time_resolver binding_times t _ _
    result Hconsistent (canonical_element_canonical t e Hconsistent)
    (canonical_element_canonical t e' Hconsistent) Hadd).
Qed.

Lemma forget_in_aliases_consistent
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (x : variable) :
  aliases_consistent t ->
  aliases_consistent (Known.forget_in_aliases binding_times t x).
Proof.
  intros Hconsistent. unfold Known.forget_in_aliases.
  destruct (negb _); [ exact Hconsistent | ]. cbv zeta.
  apply fold_left_invariant; [ exact empty_consistent | ].
  intros rebuilt n _ Hrebuilt.
  destruct (simple_equal _ _);
    [ destruct (find _ _); [ | exact Hrebuilt ] | ];
    apply add_equal_consistent, Hrebuilt.
Qed.

Theorem bind_consistent : bind_consistent_statement.
Proof.
  intros known x Hconsistent.
  exact (forget_in_aliases_consistent _ _ x Hconsistent).
Qed.

Theorem under_let_consistent : under_let_consistent_statement.
Proof.
  intros known x d Hconsistent.
  pose proof (bind_consistent known x Hconsistent) as Hbound.
  unfold Known.under_let. cbv zeta.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hbound.
  - destruct (var_in_simple x s); [ exact Hbound | ].
    cbn [Known.aliases Known.with_aliases].
    apply add_equal_consistent, Hbound.
  - destruct (var_in_value_slots x slots); exact Hbound.
  - destruct (var_in_value_slots x slots); exact Hbound.
  - destruct closure as [y | sym | k]; try exact Hbound.
    destruct (lookup y _) as [[code slots | slot decls slots] | ];
      try exact Hbound.
    destruct (Pos.eqb slot move_from); exact Hbound.
Qed.

(* A simple is in the class of its canonical element. *)
Lemma aliased_canonical (t : Aliases.t) (e : simple) :
  aliases_consistent t -> aliased t e (canonical_element_of t e).
Proof.
  intros Hconsistent. unfold aliased. symmetry.
  exact (canonical_element_idempotent t e Hconsistent).
Qed.

Lemma canonical_element_of_empty (e : simple) :
  canonical_element_of Aliases.empty e = e.
Proof. destruct e; reflexivity. Qed.

(* [add_equal] of two simples that a structure [t] knows to be equal, to
   a structure that knows no more than [t] does: the classes of the two
   are made one, and nothing else changes. *)
Lemma add_equal_classes
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t rebuilt : Aliases.t) (e e' : simple) :
  aliases_consistent rebuilt ->
  (∀ a b : simple, aliased rebuilt a b -> aliased t a b) ->
  aliased t e e' ->
  ∀ a b : simple,
    aliased (Known.add_equal binding_times rebuilt e e') a b
    <-> aliased rebuilt a b
        ∨ (aliased rebuilt a e ∧ aliased rebuilt b e')
        ∨ (aliased rebuilt a e' ∧ aliased rebuilt b e).
Proof.
  intros Hconsistent Hwithin Haliased a b. unfold Known.add_equal.
  pose proof (aliased_canonical rebuilt e Hconsistent) as H1.
  pose proof (aliased_canonical rebuilt e' Hconsistent) as H2.
  pose proof (canonical_element_canonical rebuilt e Hconsistent) as Hc1.
  pose proof (canonical_element_canonical rebuilt e' Hconsistent) as Hc2.
  destruct (simple_equal_spec (canonical_element_of rebuilt e)
              (canonical_element_of rebuilt e')) as [Heq | Hne].
  - unfold aliased in *. split; [ auto | ].
    intros [H | [[Ha Hb] | [Ha Hb]]]; congruence.
  - destruct (Aliases.add _ _ rebuilt _ _) as [[result | ] | ] eqn:Hadd.
    + rewrite (add_aliased unit Known.binding_time_resolver binding_times
                 rebuilt _ _ result a b Hconsistent Hc1 Hc2 Hadd).
      unfold aliased in *.
      split; intros [H | [[Ha Hb] | [Ha Hb]]];
        solve
          [ left; exact H
          | right; left; split; congruence
          | right; right; split; congruence ].
    + exfalso.
      apply (proj1 (add_bottom unit Known.binding_time_resolver binding_times
                      rebuilt _ _)) in Hadd.
      destruct Hadd as (k1 & k2 & Hk1 & Hk2 & Hk). apply Hk.
      rewrite Hk1 in H1. rewrite Hk2 in H2.
      apply Hwithin in H1. apply Hwithin in H2.
      unfold aliased in H1, H2, Haliased.
      assert (Hsame :
        canonical_element_of t (Const k1) = canonical_element_of t (Const k2))
        by congruence.
      injection Hsame. auto.
    + exfalso.
      exact (add_defined unit Known.binding_time_resolver binding_times
        rebuilt _ _ Hconsistent Hc1 Hc2 Hne Hadd).
Qed.

Lemma fold_left_pointwise {A B : Type} (f g : A -> B -> A) (l : list B) :
  (∀ (a : A) (b : B), f a b = g a b) ->
  ∀ init : A, List.fold_left f l init = List.fold_left g l init.
Proof.
  intros Hfg. induction l as [ | b l IH]; intros init; simpl;
    [ reflexivity | ].
  rewrite Hfg. apply IH.
Qed.

Section Forgetting.

Context
  (binding_times
    : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
  (t : Aliases.t) (x : variable).
Hypothesis t_consistent : aliases_consistent t.

(* The names that [Known.forget_in_aliases] goes through, and what
   stands for [x]. *)
Let demoted_to_x (n : name) : bool :=
  simple_equal (Aliases.get_canonical_ignoring_name_mode t n) (Var x).

Let demoted : list name :=
  filter (fun n => negb (name_eqb n (Name_var x)))
    (Keyed.domain name_eqb (Aliases.canonical_elements t)).

Let in_place_of_x : option name := find demoted_to_x demoted.

(* What each of them is made an alias of. *)
Let target (n : name) : simple :=
  if demoted_to_x n
  then
    match in_place_of_x with
    | Some n' => simple_of_name n'
    | None => simple_of_name n
    end
  else Aliases.get_canonical_ignoring_name_mode t n.

Lemma target_aliased (n : name) : aliased t (simple_of_name n) (target n).
Proof.
  unfold target.
  destruct (demoted_to_x n) eqn:Hx.
  - destruct in_place_of_x as [n' | ] eqn:Hfind; [ | reflexivity ].
    apply find_some in Hfind. destruct Hfind as [_ Hx'].
    apply simple_equal_eq in Hx. apply simple_equal_eq in Hx'.
    unfold aliased.
    rewrite <- !(get_canonical_ignoring_name_mode t). congruence.
  - rewrite (get_canonical_ignoring_name_mode t n).
    apply aliased_canonical, t_consistent.
Qed.

Lemma forget_in_aliases_eq :
  among x (aliases_free_variables t) = true ->
  Known.forget_in_aliases binding_times t x
  = List.fold_left
      (fun rebuilt n =>
         Known.add_equal binding_times rebuilt (simple_of_name n) (target n))
      demoted Aliases.empty.
Proof.
  intros Hamong. unfold Known.forget_in_aliases. rewrite Hamong.
  cbn [negb]. cbv zeta. apply fold_left_pointwise. intros rebuilt n.
  unfold target, demoted_to_x, in_place_of_x, demoted.
  destruct (simple_equal (Aliases.get_canonical_ignoring_name_mode t n)
              (Var x)); [ | reflexivity ].
  destruct (find _ _); [ reflexivity | ].
  unfold Known.add_equal.
  destruct (simple_equal_spec
              (canonical_element_of rebuilt (simple_of_name n))
              (canonical_element_of rebuilt (simple_of_name n)))
    as [_ | Hne]; [ reflexivity | contradiction ].
Qed.

Lemma rebuild_classes (l : list name) :
  ∀ rebuilt : Aliases.t,
    aliases_consistent rebuilt ->
    (∀ a b : simple, aliased rebuilt a b -> aliased t a b) ->
    let rebuilt' :=
      List.fold_left
        (fun rebuilt n =>
           Known.add_equal binding_times rebuilt (simple_of_name n)
             (target n))
        l rebuilt
    in
    (∀ a b : simple, aliased rebuilt a b -> aliased rebuilt' a b)
    ∧ ∀ n : name, n ∈ l -> aliased rebuilt' (simple_of_name n) (target n).
Proof.
  induction l as [ | n l IH]; intros rebuilt Hconsistent Hwithin; simpl.
  - split; [ auto | intros n [] ].
  - pose proof (add_equal_classes binding_times t rebuilt (simple_of_name n)
                  (target n) Hconsistent Hwithin (target_aliased n))
      as Hclasses.
    destruct (IH (Known.add_equal binding_times rebuilt (simple_of_name n)
                    (target n)))
      as [Hkept Htargets].
    + apply add_equal_consistent, Hconsistent.
    + intros a b Hab. apply Hclasses in Hab.
      pose proof (target_aliased n) as Hn. unfold aliased in *.
      destruct Hab as [H | [[Ha Hb] | [Ha Hb]]].
      * exact (Hwithin a b H).
      * pose proof (Hwithin _ _ Ha). pose proof (Hwithin _ _ Hb). congruence.
      * pose proof (Hwithin _ _ Ha). pose proof (Hwithin _ _ Hb). congruence.
    + split.
      * intros a b Hab. apply Hkept, Hclasses. left. exact Hab.
      * intros n' [<- | Hin]; [ | exact (Htargets n' Hin) ].
        apply Hkept, Hclasses. right. left. split; reflexivity.
Qed.

Lemma in_demoted (n : name) (c : simple) :
  demoted_to t n c -> simple_of_name n <> Var x -> n ∈ demoted.
Proof.
  intros Hdemoted Hne. unfold demoted. apply filter_In. split.
  - apply (keyed_among_in name name_eqb name_eqb_eq).
    rewrite (keyed_among_domain name name_eqb name_eqb_eq).
    unfold Keyed.is_bound. unfold demoted_to in Hdemoted.
    rewrite Hdemoted. reflexivity.
  - apply negb_true_iff.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq n (Name_var x))
      as [-> | _]; [ | reflexivity ].
    exfalso. apply Hne. reflexivity.
Qed.

Lemma forget_in_aliases_aliased (e e' : simple) :
  aliased t e e' ->
  e <> Var x ->
  e' <> Var x ->
  aliased (Known.forget_in_aliases binding_times t x) e e'.
Proof.
  intros Haliased He He'.
  destruct (among x (aliases_free_variables t)) eqn:Hamong.
  2: { unfold Known.forget_in_aliases. rewrite Hamong. exact Haliased. }
  rewrite (forget_in_aliases_eq Hamong).
  destruct (rebuild_classes demoted Aliases.empty empty_consistent)
    as [_ Htargets].
  { intros a b Hab. unfold aliased in Hab.
    rewrite !canonical_element_of_empty in Hab. subst b. reflexivity. }
  cbv zeta in Htargets.
  set (rebuilt :=
    List.fold_left
      (fun rebuilt n =>
         Known.add_equal binding_times rebuilt (simple_of_name n) (target n))
      demoted Aliases.empty) in *.
  (* What stands, in what is rebuilt, for a class of [t]. *)
  set (stands_for := fun c : simple =>
    if simple_equal c (Var x)
    then
      match in_place_of_x with
      | Some n' => simple_of_name n'
      | None => c
      end
    else c).
  assert (Hstands :
    ∀ s : simple,
      s <> Var x ->
      aliased rebuilt s (stands_for (canonical_element_of t s))).
  { intros s Hs.
    destruct (canonical_cases t s)
      as [[_ ->] | (n & c & -> & Hdemoted & _ & ->)].
    - unfold stands_for.
      destruct (simple_equal_spec s (Var x)) as [Heq | _];
        [ contradiction | reflexivity ].
    - pose proof (Htargets n (in_demoted n c Hdemoted Hs)) as Hn.
      assert (Htarget : target n = stands_for c).
      { unfold target, stands_for, demoted_to_x.
        rewrite (get_canonical_ignoring_name_mode t n),
          (canonical_element_of_demoted t n c Hdemoted).
        destruct (simple_equal c (Var x)) eqn:Hx; [ | reflexivity ].
        destruct in_place_of_x as [n' | ] eqn:Hfind; [ reflexivity | ].
        exfalso.
        pose proof (find_none _ _ Hfind n (in_demoted n c Hdemoted Hs))
          as Hnone.
        unfold demoted_to_x in Hnone.
        rewrite (get_canonical_ignoring_name_mode t n),
          (canonical_element_of_demoted t n c Hdemoted), Hx in Hnone.
        discriminate Hnone. }
      rewrite <- Htarget. exact Hn. }
  pose proof (Hstands e He) as H1. pose proof (Hstands e' He') as H2.
  unfold aliased in *. congruence.
Qed.

End Forgetting.

Theorem forget_aliased : forget_aliased_statement.
Proof.
  intros binding_times t x e e' Hconsistent.
  exact (forget_in_aliases_aliased binding_times t x Hconsistent e e').
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma find_renamed {A B : Type} (g : A -> B) (p : A -> bool)
    (p' : B -> bool) (l : list A) :
  (∀ a : A, p' (g a) = p a) -> find p' (map g l) = option_map g (find p l).
Proof.
  intros Hp. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hp. destruct (p a); [ reflexivity | exact IH ].
Qed.

Lemma existsb_renamed {A B : Type} (g : A -> B) (p : A -> bool)
    (p' : B -> bool) (l : list A) :
  (∀ a : A, p' (g a) = p a) -> existsb p' (map g l) = existsb p l.
Proof.
  intros Hp. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hp, IH. reflexivity.
Qed.

Lemma var_in_simple_renamed (r : renaming) (x : variable) (s : simple) :
  var_in_simple (rename_variable r x) (rename_simple r s) = var_in_simple x s.
Proof. destruct s; simpl; [ apply rename_variable_eqb | | ]; reflexivity. Qed.

Lemma var_in_value_slots_renamed (r : renaming) (x : variable)
    (slots : list (value_slot * simple)) :
  var_in_value_slots (rename_variable r x)
    (map_values (rename_simple r) slots)
  = var_in_value_slots x slots.
Proof.
  unfold var_in_value_slots, map_values. apply existsb_renamed.
  intros [slot s]. apply var_in_simple_renamed.
Qed.

Lemma canonical_element_of_renamed (r : renaming) (t : Aliases.t)
    (e : simple) :
  canonical_element_of (rename_aliases r t) (rename_simple r e)
  = rename_simple r (canonical_element_of t e).
Proof.
  unfold canonical_element_of. rewrite canonical_renamed.
  destruct (Aliases.canonical t e); reflexivity.
Qed.

Lemma canonical_simple_renamed (r : renaming) (known : Known.t) (s : simple) :
  Known.canonical_simple (rename_known r known) (rename_simple r s)
  = rename_simple r (Known.canonical_simple known s).
Proof. apply canonical_element_of_renamed. Qed.

(* The resolver answers the same of a name and of the name renamed. *)
Lemma resolver_renamed (r : renaming) (n : name) :
  Known.binding_time_resolver n
  = rename_resolver r Known.binding_time_resolver n.
Proof. destruct n; reflexivity. Qed.

Lemma add_equal_renamed (r : renaming)
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (e e' : simple) :
  Known.add_equal (rename_binding_times r binding_times) (rename_aliases r t)
    (rename_simple r e) (rename_simple r e')
  = rename_aliases r (Known.add_equal binding_times t e e').
Proof.
  unfold Known.add_equal.
  rewrite !canonical_element_of_renamed, simple_equal_renamed.
  destruct (simple_equal _ _); [ reflexivity | ].
  rewrite
    (add_alike unit unit Known.binding_time_resolver
       (rename_resolver r Known.binding_time_resolver)
       (rename_binding_times r binding_times)
       (rename_binding_times r binding_times) (rename_aliases r t) _ _
       (resolver_renamed r) (fun n => eq_refl)).
  rewrite (rename_add unit r Known.binding_time_resolver binding_times t).
  destruct (Aliases.add Known.binding_time_resolver binding_times t _ _)
    as [[result | ] | ]; reflexivity.
Qed.

Lemma demoted_to_variable_renamed (r : renaming) (t : Aliases.t)
    (x : variable) (n : name) :
  simple_equal
    (Aliases.get_canonical_ignoring_name_mode (rename_aliases r t)
       (rename_name r n))
    (Var (rename_variable r x))
  = simple_equal (Aliases.get_canonical_ignoring_name_mode t n) (Var x).
Proof.
  rewrite (rename_canonical r t n).
  exact (simple_equal_renamed r _ (Var x)).
Qed.

Lemma forget_in_aliases_renamed (r : renaming)
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (x : variable) :
  Known.forget_in_aliases (rename_binding_times r binding_times)
    (rename_aliases r t) (rename_variable r x)
  = rename_aliases r (Known.forget_in_aliases binding_times t x).
Proof.
  unfold Known.forget_in_aliases.
  assert (Hwritten :
    among (rename_variable r x)
      (aliases_free_variables (rename_aliases r t))
    = among x (aliases_free_variables t)).
  { rewrite (rename_free_variables r t).
    exact (among_keys_renamed Pos.eqb (rename_variable r)
      (rename_variable_eqb r) x (aliases_free_variables t)). }
  rewrite Hwritten.
  destruct (negb (among x (aliases_free_variables t))); [ reflexivity | ].
  cbv zeta.
  assert (Hdemoted :
    filter (fun n => negb (name_eqb n (Name_var (rename_variable r x))))
      (Keyed.domain name_eqb
         (Aliases.canonical_elements (rename_aliases r t)))
    = map (rename_name r)
        (filter (fun n => negb (name_eqb n (Name_var x)))
           (Keyed.domain name_eqb (Aliases.canonical_elements t)))).
  { change (Aliases.canonical_elements (rename_aliases r t))
      with (Keyed.map_bindings (rename_name r) (rename_simple r)
              (Aliases.canonical_elements t)).
    rewrite (domain_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
    apply filter_renamed. intros n.
    change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
    rewrite name_eqb_renamed. reflexivity. }
  rewrite Hdemoted.
  rewrite
    (find_renamed (rename_name r)
       (fun n =>
          simple_equal (Aliases.get_canonical_ignoring_name_mode t n) (Var x))
       (fun n =>
          simple_equal
            (Aliases.get_canonical_ignoring_name_mode (rename_aliases r t) n)
            (Var (rename_variable r x)))
       _ (demoted_to_variable_renamed r t x)).
  apply (fold_left_renamed (rename_aliases r) (rename_name r) _ _ _)
    with (init := Aliases.empty).
  intros rebuilt n. rewrite demoted_to_variable_renamed.
  destruct (simple_equal (Aliases.get_canonical_ignoring_name_mode t n)
              (Var x)).
  - destruct (find _ _) as [n' | ]; simpl; [ | reflexivity ].
    rewrite <- !rename_simple_of_name. apply add_equal_renamed.
  - rewrite (rename_canonical r t n), <- rename_simple_of_name.
    apply add_equal_renamed.
Qed.

Lemma closure_value_slots_renamed (r : renaming) (c : Known.closure) :
  Known.closure_value_slots (rename_closure r c)
  = map_values (rename_simple r) (Known.closure_value_slots c).
Proof. destruct c; reflexivity. Qed.

Lemma forget_renamed (r : renaming) (known : Known.t) (x : variable) :
  Known.forget (rename_known r known) (rename_variable r x)
  = rename_known r (Known.forget known x).
Proof.
  unfold Known.forget, rename_known.
  cbn [Known.aliases Known.binding_times Known.next_binding_time
       Known.closures Known.static_closures].
  f_equal.
  - apply forget_in_aliases_renamed.
  - apply filter_renamed. intros [y c]. cbn [fst snd].
    rewrite rename_variable_eqb, closure_value_slots_renamed,
      var_in_value_slots_renamed.
    reflexivity.
  - apply filter_renamed. intros [s c].
    cbn [fst snd rename_static_closure Known.as_closure].
    rewrite closure_value_slots_renamed, var_in_value_slots_renamed.
    reflexivity.
Qed.

Theorem rename_bind : rename_bind_statement.
Proof.
  intros r known x. unfold Known.bind. cbv zeta. rewrite forget_renamed.
  reflexivity.
Qed.

Lemma bind_all_renamed (r : renaming) (params : list variable) :
  ∀ known : Known.t,
    Known.bind_all (rename_known r known) (map (rename_variable r) params)
    = rename_known r (Known.bind_all known params).
Proof.
  unfold Known.bind_all.
  induction params as [ | param params IH]; intros known; simpl;
    [ reflexivity | ].
  rewrite (rename_bind r known param). apply IH.
Qed.

Lemma find_closure_renamed (r : renaming) (known : Known.t) (s : simple) :
  Known.find_closure (rename_known r known) (rename_simple r s)
  = option_map (rename_closure r) (Known.find_closure known s).
Proof.
  destruct s as [y | sym | c]; [ | | reflexivity ].
  - exact (lookup_renamed Pos.eqb (rename_variable r) (rename_variable_eqb r)
      (rename_closure r) y (Known.closures known)).
  - cbn [rename_simple Known.find_closure rename_known Known.static_closures].
    rewrite
      (lookup_map_values Known.static_closure Known.static_closure
         (rename_static_closure r) sym (Known.static_closures known)).
    destruct (lookup sym (Known.static_closures known)); reflexivity.
Qed.

(* The test of the renamed value slots, as [rename_named] writes
   them, is the test of the value slots. *)
Ltac renamed_slots_test r x slots :=
  match goal with
  | |- context [var_in_value_slots (rename_variable r x) ?renamed] =>
    replace (var_in_value_slots (rename_variable r x) renamed)
      with (var_in_value_slots x slots)
      by (symmetry; exact (var_in_value_slots_renamed r x slots))
  end.

Theorem rename_under_let : rename_under_let_statement.
Proof.
  intros r known x d. unfold Known.under_let. cbv zeta.
  rewrite (rename_bind r known x).
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    cbn [rename_named]; try reflexivity.
  - rewrite var_in_simple_renamed.
    destruct (var_in_simple x s); [ reflexivity | ].
    unfold Known.with_aliases, rename_known.
    cbn [Known.aliases Known.binding_times Known.next_binding_time
         Known.closures Known.static_closures].
    f_equal.
    exact (add_equal_renamed r _ _ (Var x) s).
  - renamed_slots_test r x slots.
    destruct (var_in_value_slots x slots); reflexivity.
  - renamed_slots_test r x slots.
    destruct (var_in_value_slots x slots); reflexivity.
  - destruct closure as [y | sym | k]; try reflexivity.
    cbn [rename_simple].
    change
      (lookup (rename_variable r y)
         (Known.closures (rename_known r (Known.bind known x))))
      with
        (Known.find_closure (rename_known r (Known.bind known x))
           (rename_simple r (Var y))).
    rewrite find_closure_renamed. cbn [Known.find_closure].
    destruct (lookup y (Known.closures (Known.bind known x)))
      as [[code slots | slot decls slots] | ]; try reflexivity.
    cbn [option_map rename_closure].
    destruct (Pos.eqb slot move_from); reflexivity.
Qed.

Theorem rename_under_let_static : rename_under_let_static_statement.
Proof.
  intros r known bindings. unfold Known.under_let_static, rename_known.
  cbn [Known.aliases Known.binding_times Known.next_binding_time
       Known.closures Known.static_closures].
  f_equal. unfold map_values. rewrite map_app. f_equal.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite map_app, IH. f_equal.
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    reflexivity.
Qed.

Lemma canonical_simples_renamed (r : renaming) (known : Known.t)
    (args : list simple) :
  map (Known.canonical_simple (rename_known r known))
    (map (rename_simple r) args)
  = map (rename_simple r) (map (Known.canonical_simple known) args).
Proof.
  rewrite !map_map. apply map_ext. intros s. apply canonical_simple_renamed.
Qed.

Lemma canonical_value_slots_renamed (r : renaming) (known : Known.t)
    (slots : list (value_slot * simple)) :
  canonical_value_slots (rename_known r known)
    (map_values (rename_simple r) slots)
  = map_values (rename_simple r) (canonical_value_slots known slots).
Proof.
  unfold canonical_value_slots, map_values. rewrite !map_map.
  apply map_ext. intros [slot s]. cbn [fst snd].
  rewrite canonical_simple_renamed. reflexivity.
Qed.

Lemma lookup_value_slot_renamed (r : renaming) (slot : value_slot)
    (c : Known.closure) :
  lookup slot (Known.closure_value_slots (rename_closure r c))
  = option_map (rename_simple r) (lookup slot (Known.closure_value_slots c)).
Proof.
  rewrite closure_value_slots_renamed.
  exact (lookup_map_values simple simple (rename_simple r) slot _).
Qed.

Lemma canonical_project_renamed (r : renaming) (known : Known.t)
    (slot : value_slot) (closure : simple) :
  canonical_project (rename_known r known) slot (rename_simple r closure)
  = rename_named r (canonical_project known slot closure).
Proof.
  unfold canonical_project. rewrite find_closure_renamed.
  destruct (Known.find_closure known closure) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite lookup_value_slot_renamed.
  destruct (lookup slot (Known.closure_value_slots c)); reflexivity.
Qed.

Lemma canonical_project_from_renamed (r : renaming) (known : Known.t)
    (project_from : function_slot) (slot : value_slot) (closure : simple) :
  canonical_project_from (rename_known r known) project_from slot
    (rename_simple r closure)
  = rename_named r (canonical_project_from known project_from slot closure).
Proof.
  unfold canonical_project_from. rewrite find_closure_renamed.
  destruct (Known.find_closure known closure) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  replace (Known.closure_slot (rename_closure r c))
    with (Known.closure_slot c) by (destruct c; reflexivity).
  destruct (Pos.eqb (Known.closure_slot c) project_from); [ | reflexivity ].
  rewrite lookup_value_slot_renamed.
  destruct (lookup slot (Known.closure_value_slots c)); reflexivity.
Qed.

Lemma canonical_project_function_slot_renamed (r : renaming)
    (known : Known.t) (move_from move_to : function_slot) (closure : simple) :
  canonical_project_function_slot (rename_known r known) move_from move_to
    (rename_simple r closure)
  = rename_named r
      (canonical_project_function_slot known move_from move_to closure).
Proof.
  unfold canonical_project_function_slot.
  destruct closure as [x | s | k]; try reflexivity.
  cbn [rename_simple rename_known Known.static_closures].
  rewrite
    (lookup_map_values Known.static_closure Known.static_closure
       (rename_static_closure r) s (Known.static_closures known)).
  destruct (lookup s (Known.static_closures known)) as [sc | ];
    cbn [option_map]; [ | reflexivity ].
  cbn [rename_static_closure Known.as_closure Known.closure_symbols].
  destruct (Known.as_closure sc) as [code slots | slot decls slots];
    cbn [rename_closure]; [ reflexivity | ].
  destruct (Pos.eqb slot move_from && is_bound move_to decls);
    [ | reflexivity ].
  destruct (lookup move_to (Known.closure_symbols sc)); reflexivity.
Qed.

Lemma canonical_named_renamed (r : renaming) (known : Known.t) (d : named) :
  canonical_named (rename_known r known) (rename_named r d)
  = rename_named r (canonical_named known d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    cbn [rename_named canonical_named].
  - rewrite canonical_simple_renamed. reflexivity.
  - rewrite canonical_simples_renamed. reflexivity.
  - f_equal. apply canonical_value_slots_renamed.
  - f_equal. apply canonical_value_slots_renamed.
  - rewrite canonical_simple_renamed. apply canonical_project_renamed.
  - rewrite canonical_simple_renamed. apply canonical_project_from_renamed.
  - rewrite canonical_simple_renamed.
    apply canonical_project_function_slot_renamed.
Qed.

Lemma canonical_static_binding_renamed (r : renaming) (known : Known.t)
    (binding : static_binding) :
  canonical_static_binding (rename_known r known)
    (rename_static_binding r binding)
  = rename_static_binding r (canonical_static_binding known binding).
Proof.
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    cbn [rename_static_binding canonical_static_binding rename_static_const
         canonical_static_const];
    try reflexivity.
  - do 2 f_equal. apply canonical_value_slots_renamed.
  - do 2 f_equal. apply canonical_value_slots_renamed.
  - rewrite canonical_simples_renamed. reflexivity.
  - rewrite canonical_simple_renamed. reflexivity.
  - rewrite canonical_simples_renamed. reflexivity.
Qed.

Lemma canonical_apply_cont_renamed (r : renaming) (known : Known.t)
    (ac : apply_cont) :
  canonical_apply_cont (rename_known r known) (rename_apply_cont r ac)
  = rename_apply_cont r (canonical_apply_cont known ac).
Proof.
  unfold canonical_apply_cont, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite canonical_simples_renamed. reflexivity.
Qed.

Lemma canonical_call_kind_renamed (r : renaming) (known : Known.t)
    (callee : simple) (kind : call_kind) :
  canonical_call_kind (rename_known r known) (rename_simple r callee) kind
  = canonical_call_kind known callee kind.
Proof.
  unfold canonical_call_kind. destruct kind as [code | ]; [ reflexivity | ].
  rewrite find_closure_renamed.
  destruct (Known.find_closure known callee) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  replace (Known.closure_code (rename_closure r c))
    with (Known.closure_code c) by (destruct c; reflexivity).
  reflexivity.
Qed.

Lemma canonical_apply_renamed (r : renaming) (known : Known.t)
    (ap : apply_expr) :
  canonical_apply (rename_known r known) (rename_apply r ap)
  = rename_apply r (canonical_apply known ap).
Proof.
  unfold canonical_apply, rename_apply. cbv zeta.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite canonical_simple_renamed, canonical_simples_renamed,
    canonical_call_kind_renamed.
  f_equal. rewrite !map_map. apply map_ext. intros [s kind]. cbn [fst snd].
  rewrite canonical_simple_renamed. reflexivity.
Qed.

Fixpoint follow_aliases_renamed (r : renaming) (e : expr) :
  ∀ known : Known.t,
    follow_aliases (rename_known r known) (rename_expr r e)
    = rename_expr r (follow_aliases known e)
with follow_aliases_in_handler_renamed (r : renaming)
    (h : continuation_handler) :
  ∀ known : Known.t,
    follow_aliases_in_handler (rename_known r known) (rename_handler r h)
    = rename_handler r (follow_aliases_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; simpl.
    + rewrite canonical_named_renamed,
        (rename_under_let r known x (canonical_named known d)),
        (follow_aliases_renamed r body).
      reflexivity.
    + assert (Hbindings :
        map (canonical_static_binding (rename_known r known))
          (map (rename_static_binding r) bindings)
        = map (rename_static_binding r)
            (map (canonical_static_binding known) bindings))
        by (rewrite !map_map; apply map_ext; intros binding;
            apply canonical_static_binding_renamed).
      rewrite Hbindings, (rename_under_let_static r known),
        (follow_aliases_renamed r body).
      reflexivity.
    + rewrite (follow_aliases_in_handler_renamed r h),
        (follow_aliases_renamed r body).
      reflexivity.
    + rewrite (follow_aliases_renamed r body). reflexivity.
    + rewrite canonical_apply_renamed. reflexivity.
    + rewrite canonical_apply_cont_renamed. reflexivity.
    + rewrite canonical_simple_renamed. f_equal.
      rewrite !map_map. apply map_ext. intros [i ac]. cbn [fst snd].
      rewrite canonical_apply_cont_renamed. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. intros known. simpl.
    rewrite bind_all_renamed, (follow_aliases_renamed r handler).
    reflexivity.
Qed.

Theorem rename_follow_aliases : rename_follow_aliases_statement.
Proof. intros r known e. apply follow_aliases_renamed. Qed.

Theorem rename_follow_aliases_empty : rename_follow_aliases_empty_statement.
Proof. intros r e. exact (follow_aliases_renamed r e Known.empty). Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Definition x : variable := 1%positive.
Definition y : variable := 2%positive.
Definition z : variable := 3%positive.
Definition f : variable := 4%positive.
Definition g : variable := 5%positive.

Definition three : simple := Const (Tagged_immediate 3).

(* A jump to the continuation of the term, with arguments. *)
Definition jump (args : list simple) : expr :=
  Apply_cont (Apply_cont_expr 1%positive args None).

(* A variable bound to a constant: its uses become the constant. *)
Example constant_followed :
  follow_aliases Known.empty (Let x Value (Simple three) (jump [Var x]))
  = Let x Value (Simple three) (jump [three]).
Proof. reflexivity. Qed.

(* A chain of aliases of a variable that the term does not bind: every
   use becomes that variable, which is earlier than those bound here. *)
Example chain_followed :
  follow_aliases Known.empty
    (Let y Value (Simple (Var x))
       (Let z Value (Simple (Var y)) (jump [Var z; Var y])))
  = Let y Value (Simple (Var x))
      (Let z Value (Simple (Var x)) (jump [Var x; Var x])).
Proof. reflexivity. Qed.

(* [x] is bound again where [y] and [z] are known to be aliases of it.
   What is known of the old [x] is forgotten; that [y] and [z] are
   equal is kept, with [y], the earlier, as the canonical element. *)
Example bound_again :
  follow_aliases Known.empty
    (Let y Value (Simple (Var x))
       (Let z Value (Simple (Var x))
          (Let x Value (Simple three) (jump [Var x; Var y; Var z]))))
  = Let y Value (Simple (Var x))
      (Let z Value (Simple (Var x))
         (Let x Value (Simple three) (jump [three; Var y; Var y]))).
Proof. reflexivity. Qed.

(* A set of two closures, bound as the compiler's [Let] of a set is: the
   first closure, and the second as its projection from the first. *)
Definition f_slot : function_slot := 1%positive.
Definition g_slot : function_slot := 2%positive.
Definition f_code : code_id := 1%positive.
Definition g_code : code_id := 2%positive.
Definition v_slot : value_slot := 1%positive.

Definition two_closures (body : expr) : expr :=
  Let f Value
    (Closure_of_set f_slot [(f_slot, f_code); (g_slot, g_code)]
       [(v_slot, three)])
    (Let g Value (Project_function_slot f_slot g_slot (Var f)) body).

Definition call (callee : simple) (kind : call_kind) (args : list simple)
    : expr :=
  Apply
    (Apply_expr callee kind args 1%positive 2%positive []
       (arity_of_kinds [Value]) (arity_of_kinds [Value])).

(* A call of the second closure is a direct call of its code, and the
   value slot projected from it, at its own function slot, is what the
   slot was given. *)
Example second_closure_known :
  follow_aliases Known.empty
    (two_closures
       (Let x Value (Project_value_slot_from g_slot v_slot (Var g))
          (call (Var g) Indirect [Var x])))
  = two_closures
      (Let x Value (Simple three) (call (Var g) (Direct g_code) [three])).
Proof. reflexivity. Qed.

(* A projection that names another function slot than the one its
   closure is at is undefined, and is left as it is. *)
Example wrong_slot_left :
  follow_aliases Known.empty
    (two_closures
       (Let x Value (Project_value_slot_from f_slot v_slot (Var g))
          (jump [Var x])))
  = two_closures
      (Let x Value (Project_value_slot_from f_slot v_slot (Var g))
         (jump [Var x])).
Proof. reflexivity. Qed.

(* A static set of two closures, bound to the symbols [s_f] and [s_g],
   with a value slot that holds the variable [y]. *)
Definition s_f : symbol := 1%positive.
Definition s_g : symbol := 2%positive.

Definition two_static_closures (body : expr) : expr :=
  Let_static
    [Bind_symbol s_f
       (Static_const_closure_of_set f_slot [(f_slot, s_f); (g_slot, s_g)]
          [(f_slot, f_code); (g_slot, g_code)] [(v_slot, Var y)]);
     Bind_symbol s_g
       (Static_const_closure_of_set g_slot [(f_slot, s_f); (g_slot, s_g)]
          [(f_slot, f_code); (g_slot, g_code)] [(v_slot, Var y)])]
    body.

(* In the body of the [Let_static]: a variable bound to the first
   symbol is the symbol; the projection of the other function slot from
   it is the second symbol; a call of that is a direct call of the code
   of the second closure; and the value slot projected from it is the
   variable that the slot was given. *)
Example static_closures_known :
  follow_aliases Known.empty
    (two_static_closures
       (Let f Value (Simple (Symbol s_f))
          (Let g Value (Project_function_slot f_slot g_slot (Var f))
             (Let x Value (Project_value_slot_from g_slot v_slot (Var g))
                (call (Var g) Indirect [Var x])))))
  = two_static_closures
      (Let f Value (Simple (Symbol s_f))
         (Let g Value (Simple (Symbol s_g))
            (Let x Value (Simple (Var y))
               (call (Symbol s_g) (Direct g_code) [Var y])))).
Proof. reflexivity. Qed.

(* The variable that the value slot holds is bound again: what was
   recorded of the two symbols said what the slot holds, and is
   forgotten whole. *)
Example static_closures_forgotten :
  follow_aliases Known.empty
    (two_static_closures
       (Let y Value (Simple three)
          (Let x Value (Project_value_slot_from f_slot v_slot (Symbol s_f))
             (call (Symbol s_f) Indirect [Var x]))))
  = two_static_closures
      (Let y Value (Simple three)
         (Let x Value (Project_value_slot_from f_slot v_slot (Symbol s_f))
            (call (Symbol s_f) Indirect [Var x]))).
Proof. reflexivity. Qed.

(* Before the [Let_static] nothing is known of its symbols. *)
Example static_closures_in_scope :
  follow_aliases Known.empty
    (Let x Value (Project_function_slot f_slot g_slot (Symbol s_f))
       (two_static_closures (call (Var x) Indirect [])))
  = Let x Value (Project_function_slot f_slot g_slot (Symbol s_f))
      (two_static_closures (call (Var x) Indirect [])).
Proof. reflexivity. Qed.

(* What is learnt in the body of a [Let_cont] is not known in the
   handler, and the parameter of the handler is a new variable. *)
Example handler_apart :
  follow_aliases Known.empty
    (Let y Value (Simple three)
       (Let_cont_non_recursive 2%positive
          (Handler [z] (make_handler_metadata [Value]) (jump [Var y; Var z]))
          (Let z Value (Simple (Var y)) (jump [Var z]))))
  = Let y Value (Simple three)
      (Let_cont_non_recursive 2%positive
         (Handler [z] (make_handler_metadata [Value]) (jump [three; Var z]))
         (Let z Value (Simple three) (jump [three]))).
Proof. reflexivity. Qed.

End Examples.
