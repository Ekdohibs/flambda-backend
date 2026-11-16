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

(* Proof of the statement of FlambdaEnvAliasesFollowScopeSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNormaliseSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaEnvAliasesRenamingSpec FlambdaEnvAliasesRenamingProofs
  FlambdaEnvAliasesFollowSpec FlambdaEnvAliasesFollowProofs
  FlambdaEnvAliasesFollowScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known speaks of some variables only                              *)
(* ======================================================================== *)

(* The variables that what is known speaks of are among [names]: the
   variables of the aliases, and those of the value slots of the
   closures recorded. *)
Definition known_within (names : list variable) (known : Known.t) : Prop :=
  (∀ x : variable,
    aliases_mention (Known.aliases known) (Name_var x) -> In x names)
  ∧ (∀ entry : variable * Known.closure,
      In entry (Known.closures known) ->
      incl (free_variables_of_value_slots
              (Known.closure_value_slots (snd entry)))
        names)
  ∧ ∀ entry : symbol * Known.static_closure,
      In entry (Known.static_closures known) ->
      incl
        (free_variables_of_value_slots
           (Known.closure_value_slots (Known.as_closure (snd entry))))
        names.

Lemma known_within_weaken (names names' : list variable) (known : Known.t) :
  incl names names' -> known_within names known -> known_within names' known.
Proof.
  intros Hincl (Haliases & Hclosures & Hstatics).
  split; [ | split ].
  - intros x Hx. exact (Hincl x (Haliases x Hx)).
  - intros entry Hentry.
    exact (incl_tran (Hclosures entry Hentry) Hincl).
  - intros entry Hentry.
    exact (incl_tran (Hstatics entry Hentry) Hincl).
Qed.

Lemma empty_within : known_within [] Known.empty.
Proof.
  split; [ | split ].
  - intros x Hx. destruct (empty_unmentioned (Name_var x) Hx).
  - intros entry [].
  - intros entry [].
Qed.

(* A name that the aliases demote. *)
Lemma domain_demoted (t : Aliases.t) (n : name) :
  In n (Keyed.domain name_eqb (Aliases.canonical_elements t)) ->
  ∃ c : simple, demoted_to t n c.
Proof.
  intros Hin. apply (keyed_among_in name name_eqb name_eqb_eq) in Hin.
  rewrite (keyed_among_domain name name_eqb name_eqb_eq) in Hin.
  unfold Keyed.is_bound in Hin. unfold demoted_to.
  destruct (Keyed.lookup name_eqb n (Aliases.canonical_elements t))
    as [c | ]; [ exists c; reflexivity | discriminate Hin ].
Qed.

(* What is left of the aliases when a variable is forgotten speaks of
   names that they spoke of. *)
Lemma forget_in_aliases_mentioned
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t) (x : variable) (n : name) :
  aliases_mention (Known.forget_in_aliases binding_times t x) n ->
  aliases_mention t n.
Proof.
  unfold Known.forget_in_aliases.
  destruct (negb (among x (aliases_free_variables t)));
    [ exact (fun Hmentioned => Hmentioned) | ].
  cbv zeta. revert n.
  apply (fold_left_invariant
           (fun rebuilt : Aliases.t =>
              ∀ n : name, aliases_mention rebuilt n -> aliases_mention t n)).
  - intros n Hn. destruct (empty_unmentioned n Hn).
  - intros rebuilt n0 Hin Hrebuilt n.
    apply filter_In in Hin. destruct Hin as [Hin _].
    destruct (domain_demoted t n0 Hin) as [c0 Hdemoted0].
    assert (Hn0 : aliases_mention t n0) by (left; exists c0; exact Hdemoted0).
    assert (Hcanonical :
      simple_of_name n = Aliases.get_canonical_ignoring_name_mode t n0 ->
      aliases_mention t n).
    { intros Heq.
      destruct (canonical_element_mentioned t n0 n (eq_sym Heq))
        as [-> | Hmentioned]; assumption. }
    destruct (simple_equal (Aliases.get_canonical_ignoring_name_mode t n0)
                (Var x)).
    + destruct (find _ _) as [n' | ] eqn:Hfind; [ | apply Hrebuilt ].
      apply find_some in Hfind. destruct Hfind as [Hin' _].
      apply filter_In in Hin'. destruct Hin' as [Hin' _].
      destruct (domain_demoted t n' Hin') as [c' Hdemoted'].
      intros Hmentioned. apply add_equal_mentioned in Hmentioned.
      destruct Hmentioned as [H | [H | H]]; [ exact (Hrebuilt n H) | | ].
      * apply simple_of_name_inj in H. subst n. exact Hn0.
      * apply simple_of_name_inj in H. subst n. left. exists c'.
        exact Hdemoted'.
    + intros Hmentioned. apply add_equal_mentioned in Hmentioned.
      destruct Hmentioned as [H | [H | H]]; [ exact (Hrebuilt n H) | | ].
      * apply simple_of_name_inj in H. subst n. exact Hn0.
      * exact (Hcanonical H).
Qed.

Lemma forget_within (names : list variable) (known : Known.t) (x : variable) :
  known_within names known -> known_within names (Known.forget known x).
Proof.
  intros (Haliases & Hclosures & Hstatics). split; [ | split ].
  - intros y Hy. apply Haliases.
    exact (forget_in_aliases_mentioned _ _ x (Name_var y) Hy).
  - intros entry Hentry. cbn [Known.forget Known.closures] in Hentry.
    apply filter_In in Hentry. exact (Hclosures entry (proj1 Hentry)).
  - intros entry Hentry. cbn [Known.forget Known.static_closures] in Hentry.
    apply filter_In in Hentry. exact (Hstatics entry (proj1 Hentry)).
Qed.

Lemma bind_within (names : list variable) (known : Known.t) (x : variable) :
  known_within names known -> known_within names (Known.bind known x).
Proof.
  intros Hknown. exact (forget_within names known x Hknown).
Qed.

Lemma bind_all_within (names : list variable) (xs : list variable) :
  ∀ known : Known.t,
    known_within names known -> known_within names (Known.bind_all known xs).
Proof.
  unfold Known.bind_all.
  induction xs as [ | x xs IH]; intros known Hknown; cbn [fold_left];
    [ exact Hknown | apply IH, bind_within, Hknown ].
Qed.

(* A closure is recorded for a variable. *)
Lemma add_closure_within (names : list variable) (known : Known.t)
    (x : variable) (c : Known.closure) :
  incl (free_variables_of_value_slots (Known.closure_value_slots c)) names ->
  known_within names known ->
  known_within names (Known.add_closure known x c).
Proof.
  intros Hc (Haliases & Hclosures & Hstatics). split; [ | split ].
  - exact Haliases.
  - intros entry [<- | Hentry]; [ exact Hc | exact (Hclosures entry Hentry) ].
  - exact Hstatics.
Qed.

Lemma under_let_within (names : list variable) (known : Known.t)
    (x : variable) (d : named) :
  known_within names known ->
  known_within (x :: names ++ free_variables_of_named d)
    (Known.under_let known x d).
Proof.
  intros Hknown.
  assert (Hbound :
    known_within (x :: names ++ free_variables_of_named d)
      (Known.bind known x)).
  { apply (known_within_weaken names); [ | apply bind_within, Hknown ].
    apply incl_tl, incl_appl, incl_refl. }
  assert (Hd :
    incl (free_variables_of_named d)
      (x :: names ++ free_variables_of_named d)).
  { apply incl_tl, incl_appr, incl_refl. }
  unfold Known.under_let.
  destruct d as [s | p args | code slots | slot decls slots
                | slot closure | project_from slot closure
                | move_from move_to closure];
    try exact Hbound.
  - destruct (var_in_simple x s); [ exact Hbound | ].
    destruct Hbound as (Haliases & Hclosures & Hstatics).
    split; [ | split ]; [ | exact Hclosures | exact Hstatics ].
    intros y Hy. cbn [Known.with_aliases Known.aliases] in Hy.
    apply add_equal_mentioned in Hy.
    destruct Hy as [Hy | [Hy | Hy]].
    + exact (Haliases y Hy).
    + injection Hy as ->. left. reflexivity.
    + apply Hd. cbn [free_variables_of_named]. rewrite <- Hy. left.
      reflexivity.
  - destruct (var_in_value_slots x slots); [ exact Hbound | ].
    apply add_closure_within; [ exact Hd | exact Hbound ].
  - destruct (var_in_value_slots x slots); [ exact Hbound | ].
    apply add_closure_within; [ exact Hd | exact Hbound ].
  - destruct closure as [y | sym | c]; try exact Hbound.
    destruct (lookup y (Known.closures (Known.bind known x)))
      as [[code slots | slot decls slots] | ] eqn:Hlookup;
      try exact Hbound.
    destruct (Pos.eqb slot move_from); [ | exact Hbound ].
    apply add_closure_within; [ | exact Hbound ].
    destruct Hbound as (_ & Hclosures & _).
    exact (Hclosures (y, Known.Of_set slot decls slots)
             (lookup_in y _ _ Hlookup)).
Qed.

(* The closures that a group binds symbols to. *)
Lemma static_closure_of_binding_within (binding : static_binding)
    (entry : symbol * Known.static_closure) :
  Known.static_closure_of_binding binding = Some entry ->
  incl
    (free_variables_of_value_slots
       (Known.closure_value_slots (Known.as_closure (snd entry))))
    (free_variables_of_binding binding).
Proof.
  destruct binding as [id c | id | s const]; try discriminate.
  destruct const as [code slots | slot syms decls slots | tag fields
                    | contents | bytes | elements];
    try discriminate; intros [= <-]; apply incl_refl.
Qed.

Lemma under_let_static_within (names : list variable) (known : Known.t)
    (bindings : list static_binding) :
  known_within names known ->
  known_within (names ++ flat_map free_variables_of_binding bindings)
    (Known.under_let_static known bindings).
Proof.
  intros (Haliases & Hclosures & Hstatics). split; [ | split ].
  - intros x Hx. apply in_or_app. left. exact (Haliases x Hx).
  - intros entry Hentry. apply incl_appl. exact (Hclosures entry Hentry).
  - intros entry Hentry.
    cbn [Known.under_let_static Known.static_closures] in Hentry.
    apply in_app_or in Hentry. destruct Hentry as [Hentry | Hentry];
      [ | apply incl_appl; exact (Hstatics entry Hentry) ].
    apply in_flat_map in Hentry. destruct Hentry as [binding [Hin Hentry]].
    destruct (Known.static_closure_of_binding binding) as [entry' | ]
      eqn:Hbinding; [ | destruct Hentry ].
    destruct Hentry as [<- | []].
    apply incl_appr. intros y Hy. apply in_flat_map. exists binding.
    split; [ exact Hin | ].
    exact (static_closure_of_binding_within binding entry' Hbinding y Hy).
Qed.

(* ======================================================================== *)
(* What is put in a term                                                    *)
(* ======================================================================== *)

Section Canonical.

Variables (names : list variable) (known : Known.t).

Hypothesis within : known_within names known.

Lemma canonical_simple_free (s : simple) :
  incl (free_variables_of_simple (Known.canonical_simple known s))
    (free_variables_of_simple s ++ names).
Proof.
  destruct within as (Haliases & _). unfold Known.canonical_simple.
  destruct (canonical_element_of (Known.aliases known) s) as [y | sym | c]
    eqn:Hcanonical;
    cbn [free_variables_of_simple]; [ | apply incl_nil_l | apply incl_nil_l ].
  destruct (mentioned_canonical_element_of (Known.aliases known) s
              (Name_var y) Hcanonical)
    as [Hmentioned | Heq].
  - intros a [<- | []]. apply in_or_app. right. exact (Haliases y Hmentioned).
  - cbn [simple_of_name] in Heq. subst s. apply incl_appl, incl_refl.
Qed.

Lemma canonical_simples_free (args : list simple) :
  incl
    (free_variables_of_simples (map (Known.canonical_simple known) args))
    (free_variables_of_simples args ++ names).
Proof. apply incl_flat_map_under, canonical_simple_free. Qed.

Lemma canonical_value_slots_free (slots : list (value_slot * simple)) :
  incl (free_variables_of_value_slots (canonical_value_slots known slots))
    (free_variables_of_value_slots slots ++ names).
Proof.
  apply
    (incl_flat_map_under (fun slot => free_variables_of_simple (snd slot))
       (fun slot => free_variables_of_simple (snd slot))).
  intros slot. cbn [snd]. apply canonical_simple_free.
Qed.

(* The value slots of a closure that is known. *)
Lemma find_closure_within (s : simple) (c : Known.closure) :
  Known.find_closure known s = Some c ->
  incl (free_variables_of_value_slots (Known.closure_value_slots c)) names.
Proof.
  destruct within as (_ & Hclosures & Hstatics).
  destruct s as [x | sym | const]; cbn [Known.find_closure].
  - intros Hlookup. exact (Hclosures (x, c) (lookup_in x c _ Hlookup)).
  - destruct (lookup sym (Known.static_closures known)) as [sc | ] eqn:Hlookup;
      [ | discriminate ].
    intros [= <-]. exact (Hstatics (sym, sc) (lookup_in sym sc _ Hlookup)).
  - discriminate.
Qed.

(* What a value slot of a known closure holds. *)
Lemma slot_contents_within (c : Known.closure) (slot : value_slot)
    (contents : simple) :
  incl (free_variables_of_value_slots (Known.closure_value_slots c)) names ->
  lookup slot (Known.closure_value_slots c) = Some contents ->
  incl (free_variables_of_simple contents) names.
Proof.
  intros Hslots Hslot a Ha. apply Hslots.
  unfold free_variables_of_value_slots. apply in_flat_map.
  exists (slot, contents). split; [ exact (lookup_in _ _ _ Hslot) | exact Ha ].
Qed.

Lemma canonical_project_free (slot : value_slot) (closure : simple) :
  incl (free_variables_of_named (canonical_project known slot closure))
    (free_variables_of_simple closure ++ names).
Proof.
  unfold canonical_project.
  destruct (Known.find_closure known closure) as [c | ] eqn:Hclosure;
    [ | apply incl_appl, incl_refl ].
  destruct (lookup slot (Known.closure_value_slots c)) as [contents | ]
    eqn:Hslot;
    [ | apply incl_appl, incl_refl ].
  cbn [free_variables_of_named]. apply incl_appr.
  exact (slot_contents_within c slot contents
           (find_closure_within closure c Hclosure) Hslot).
Qed.

Lemma canonical_project_from_free (project_from : function_slot)
    (slot : value_slot) (closure : simple) :
  incl
    (free_variables_of_named
       (canonical_project_from known project_from slot closure))
    (free_variables_of_simple closure ++ names).
Proof.
  unfold canonical_project_from.
  destruct (Known.find_closure known closure) as [c | ] eqn:Hclosure;
    [ | apply incl_appl, incl_refl ].
  destruct (Pos.eqb (Known.closure_slot c) project_from);
    [ | apply incl_appl, incl_refl ].
  destruct (lookup slot (Known.closure_value_slots c)) as [contents | ]
    eqn:Hslot;
    [ | apply incl_appl, incl_refl ].
  cbn [free_variables_of_named]. apply incl_appr.
  exact (slot_contents_within c slot contents
           (find_closure_within closure c Hclosure) Hslot).
Qed.

Lemma canonical_project_function_slot_free (move_from move_to : function_slot)
    (closure : simple) :
  incl
    (free_variables_of_named
       (canonical_project_function_slot known move_from move_to closure))
    (free_variables_of_simple closure ++ names).
Proof.
  unfold canonical_project_function_slot.
  destruct closure as [x | s | c]; try (apply incl_appl, incl_refl).
  destruct (lookup s (Known.static_closures known)) as [sc | ];
    [ | apply incl_appl, incl_refl ].
  destruct (Known.as_closure sc) as [code slots | slot decls slots];
    [ apply incl_appl, incl_refl | ].
  destruct (Pos.eqb slot move_from && is_bound move_to decls);
    [ | apply incl_appl, incl_refl ].
  destruct (lookup move_to (Known.closure_symbols sc)) as [s' | ];
    [ apply incl_nil_l | apply incl_appl, incl_refl ].
Qed.

Lemma canonical_named_free (d : named) :
  incl (free_variables_of_named (canonical_named known d))
    (free_variables_of_named d ++ names).
Proof.
  assert (Hthrough :
    ∀ (closure : simple) (d' : named),
      incl (free_variables_of_named d')
        (free_variables_of_simple (Known.canonical_simple known closure)
         ++ names) ->
      incl (free_variables_of_named d')
        (free_variables_of_simple closure ++ names)).
  { intros closure d' Hd' y Hy. apply Hd', in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ exact (canonical_simple_free closure y Hy)
      | apply in_or_app; right; exact Hy ]. }
  destruct d; cbn [canonical_named free_variables_of_named];
    try
      first
        [ apply canonical_simple_free | apply canonical_simples_free
        | apply canonical_value_slots_free ];
    apply Hthrough;
    first
      [ apply canonical_project_free | apply canonical_project_from_free
      | apply canonical_project_function_slot_free ].
Qed.

Lemma canonical_static_const_free (const : static_const) :
  incl
    (free_variables_of_static_const (canonical_static_const known const))
    (free_variables_of_static_const const ++ names).
Proof.
  destruct const; cbn [canonical_static_const free_variables_of_static_const];
    first
      [ apply canonical_simple_free | apply canonical_simples_free
      | apply canonical_value_slots_free | apply incl_appl, incl_refl ].
Qed.

Lemma canonical_static_bindings_scope (bindings : list static_binding) :
  (forallb binding_code_closed bindings = true ->
   forallb binding_code_closed
     (map (canonical_static_binding known) bindings)
   = true)
  ∧ incl
      (flat_map free_variables_of_binding
         (map (canonical_static_binding known) bindings))
      (flat_map free_variables_of_binding bindings ++ names)
  ∧ incl
      (flat_map free_continuations_of_binding
         (map (canonical_static_binding known) bindings))
      (flat_map free_continuations_of_binding bindings).
Proof.
  split; [ | split ].
  - induction bindings as [ | binding bindings IH]; cbn [map forallb];
      [ intros _; reflexivity | ].
    intros Hclosed. apply andb_true_iff in Hclosed.
    destruct Hclosed as [Hfirst Hrest]. rewrite (IH Hrest), andb_true_r.
    destruct binding; exact Hfirst.
  - apply incl_flat_map_under. intros binding.
    destruct binding as [id c | id | sym const];
      cbn [canonical_static_binding free_variables_of_binding];
      [ apply incl_appl, incl_refl | apply incl_appl, incl_refl
      | apply canonical_static_const_free ].
  - rewrite
      (flat_map_map_same free_continuations_of_binding
         free_continuations_of_binding);
      [ apply incl_refl | ].
    intros binding. destruct binding; reflexivity.
Qed.

Lemma canonical_apply_cont_free (ac : apply_cont) :
  incl (free_variables_of_apply_cont (canonical_apply_cont known ac))
    (free_variables_of_apply_cont ac ++ names).
Proof.
  unfold free_variables_of_apply_cont, canonical_apply_cont. cbn [ac_args].
  apply canonical_simples_free.
Qed.

Lemma exn_extra_simples_canonical (ap : apply_expr) :
  exn_extra_simples (canonical_apply known ap)
  = map (Known.canonical_simple known) (exn_extra_simples ap).
Proof.
  unfold exn_extra_simples, canonical_apply. cbn [ap_exn_extra_args].
  rewrite !map_map. reflexivity.
Qed.

Lemma canonical_apply_free (ap : apply_expr) :
  incl (free_variables_of_apply (canonical_apply known ap))
    (free_variables_of_apply ap ++ names).
Proof.
  unfold free_variables_of_apply. rewrite exn_extra_simples_canonical.
  unfold canonical_apply. cbn [ap_callee ap_args].
  apply incl_app_under; [ apply canonical_simple_free | ].
  apply incl_app_under; apply canonical_simples_free.
Qed.

End Canonical.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

Fixpoint follow_aliases_scoped_under (e : expr) :
  ∀ (names : list variable) (known : Known.t),
    known_within names known ->
    scoped_under names e (follow_aliases known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [follow_aliases follow_aliases_in_handler].
  - apply scoped_under_let; [ apply canonical_named_free, Hknown | ].
    apply follow_aliases_scoped_under.
    eapply known_within_weaken; [ | apply under_let_within, Hknown ].
    intros y [<- | Hy]; [ left; reflexivity | right ].
    apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | ].
    apply (canonical_named_free names known Hknown d), in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ right; exact Hy | left; exact Hy ].
  - destruct (canonical_static_bindings_scope names known Hknown bindings)
      as (Hcode & Hv & Hk).
    apply scoped_under_let_static_bindings;
      [ exact Hcode | exact Hv | exact Hk | ].
    apply follow_aliases_scoped_under.
    eapply known_within_weaken; [ | apply under_let_static_within, Hknown ].
    intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | ].
    apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ right; exact Hy | left; exact Hy ].
  - apply scoped_under_let_cont; apply follow_aliases_scoped_under;
      [ apply bind_all_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply follow_aliases_scoped_under, Hknown.
  - intros _. scope.
    split; [ reflexivity | split; [ | apply incl_refl ] ].
    apply canonical_apply_free, Hknown.
  - intros _. scope.
    split; [ reflexivity | split; [ | apply incl_refl ] ].
    apply canonical_apply_cont_free, Hknown.
  - intros _. scope. split; [ reflexivity | split ].
    + apply incl_app_under; [ apply canonical_simple_free, Hknown | ].
      apply
        (incl_flat_map_under
           (fun arm => free_variables_of_apply_cont (snd arm))
           (fun arm => free_variables_of_apply_cont (snd arm))).
      intros arm. cbn [snd]. apply canonical_apply_cont_free, Hknown.
    + rewrite
        (flat_map_map_same (fun arm => apply_cont_continuations (snd arm))
           (fun arm => apply_cont_continuations (snd arm)));
        [ apply incl_refl | intros arm; reflexivity ].
  - apply scoped_under_refl.
Qed.

Theorem follow_aliases_knowing_nothing_keeps_scope
    : follow_aliases_knowing_nothing_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e.
  apply scoped_under_none, follow_aliases_scoped_under, empty_within.
Qed.
