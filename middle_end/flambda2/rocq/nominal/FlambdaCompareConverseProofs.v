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

(* Proofs of the statements of FlambdaCompareConverseSpec.v. Nothing
   here needs review.

   The recursion is on the first term, and the statement proved is of
   the term under any renaming, since at a binder the scope is renamed
   before it is compared again. Each case is a lemma that takes what
   the recursion gives for the parts of the term; the recursion itself
   is the last proof of the file.

   At a binder: take fresh names for both sides ([fresh_opens_at]);
   the scopes, with the bound names of each side exchanged for them,
   are accepted with no names paired ([opened_eqb]: the comparison
   accepts each scope against its renaming, and is symmetric and
   transitive under maps that compose); the recursion gives that they
   are alpha-equivalent; and the rule that compares two binders at the
   same names concludes. Where there is no binder the comparison with
   no names paired is equality. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaBindersSpec
  FlambdaBindersProofs FlambdaFreeNamesSpec FlambdaFreeNamesProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaCompareSpec FlambdaCompareProofs
  FlambdaCompareAlphaSpec FlambdaCompareAlphaProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaFreshQuantifierSpec FlambdaFreshQuantifierProofs
  FlambdaFreshBindersSpec FlambdaFreshBindersProofs
  FlambdaCompareConverseSpec.
Import ListNotations.

(* ======================================================================== *)
(* With no names paired, the comparison of what has no binder is equality   *)
(* ======================================================================== *)

Lemma forall2b_eq {A : Type} (f : A -> A -> bool) :
  (∀ a b : A, f a b = true -> a = b) ->
  ∀ l l' : list A, forall2b f l l' = true -> l = l'.
Proof.
  intros Hf. induction l as [ | a l IH]; intros [ | b l']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hab Hl].
  rewrite (Hf a b Hab), (IH l' Hl). reflexivity.
Qed.

Lemma simple_eqb_nil (s s' : simple) : simple_eqb [] s s' = true -> s = s'.
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl;
    try discriminate; intros Heq.
  - apply Pos.eqb_eq in Heq. congruence.
  - apply Pos.eqb_eq in Heq. congruence.
  - apply const_eqb_eq in Heq. congruence.
Qed.

Lemma simples_eqb_nil (args args' : list simple) :
  simples_eqb [] args args' = true -> args = args'.
Proof. apply forall2b_eq. exact simple_eqb_nil. Qed.

Lemma value_slots_eqb_nil (slots slots' : list (value_slot * simple)) :
  value_slots_eqb [] slots slots' = true -> slots = slots'.
Proof.
  apply forall2b_eq. intros [slot s] [slot' s'] Heq. simpl in Heq.
  apply andb_true_iff in Heq as [Hslot Hs]. apply Pos.eqb_eq in Hslot.
  apply simple_eqb_nil in Hs. congruence.
Qed.

Lemma bytes_eqb_eq (bytes bytes' : list Z) :
  forall2b Z.eqb bytes bytes' = true -> bytes = bytes'.
Proof. apply forall2b_eq. intros a b. apply Z.eqb_eq. Qed.

Lemma raise_kind_eqb_eq (kind kind' : option raise_kind) :
  raise_kind_eqb kind kind' = true -> kind = kind'.
Proof.
  destruct kind as [[ | | ] | ], kind' as [[ | | ] | ]; simpl;
    try discriminate; reflexivity.
Qed.

Lemma trap_action_eqb_nil (action action' : option trap_action) :
  trap_action_eqb [] action action' = true -> action = action'.
Proof.
  destruct action as [[k | k kind] | ], action' as [[k' | k' kind'] | ];
    simpl; try discriminate; intros Heq.
  - apply names_match_nil in Heq. congruence.
  - apply andb_true_iff in Heq as [Hk Hkind]. apply names_match_nil in Hk.
    apply raise_kind_eqb_eq in Hkind. congruence.
  - reflexivity.
Qed.

(* Two lists of pairs with the same first components and the same
   second components. *)
Lemma pairs_eq {A B : Type} (l : list (A * B)) :
  ∀ l' : list (A * B),
    map fst l = map fst l' -> map snd l = map snd l' -> l = l'.
Proof.
  induction l as [ | [a b] l IH]; intros [ | [a' b'] l']; simpl;
    try discriminate; [ reflexivity | ].
  intros [= -> Hfst] [= -> Hsnd]. rewrite (IH l' Hfst Hsnd). reflexivity.
Qed.

(* Read each test of a conjunction back as an equation. *)
Ltac tests_to_equations :=
  repeat match goal with
    | H : _ && _ = true |- _ => apply andb_true_iff in H as [? ?]
    | H : Pos.eqb _ _ = true |- _ => apply Pos.eqb_eq in H
    | H : Z.eqb _ _ = true |- _ => apply Z.eqb_eq in H
    | H : names_match [] _ _ = true |- _ => apply names_match_nil in H
    | H : prim_eqb _ _ = true |- _ => apply prim_eqb_sound in H
    | H : names_eqb _ _ = true |- _ => apply names_eqb_eq in H
    | H : call_kind_eqb _ _ = true |- _ => apply call_kind_eqb_eq in H
    | H : simple_eqb [] _ _ = true |- _ => apply simple_eqb_nil in H
    | H : simples_eqb [] _ _ = true |- _ => apply simples_eqb_nil in H
    | H : value_slots_eqb [] _ _ = true |- _ =>
      apply value_slots_eqb_nil in H
    | H : forall2b Z.eqb _ _ = true |- _ => apply bytes_eqb_eq in H
    | H : arity_eqb _ _ = true |- _ => apply arity_eqb_eq in H
    | H : kinds_eqb _ _ = true |- _ => apply kinds_eqb_eq in H
    | H : trap_action_eqb [] _ _ = true |- _ =>
      apply trap_action_eqb_nil in H
    end.

Lemma named_eqb_nil (d d' : named) : named_eqb [] d d' = true -> d = d'.
Proof.
  destruct d, d'; simpl; try discriminate; intros Heq; tests_to_equations;
    congruence.
Qed.

Lemma static_const_eqb_nil (const const' : static_const) :
  static_const_eqb [] const const' = true -> const = const'.
Proof.
  destruct const, const'; simpl; try discriminate; intros Heq;
    tests_to_equations; congruence.
Qed.

Lemma apply_cont_eqb_nil (ac ac' : apply_cont) :
  apply_cont_eqb [] [] ac ac' = true -> ac = ac'.
Proof.
  destruct ac, ac'. unfold apply_cont_eqb. simpl. intros Heq.
  tests_to_equations. congruence.
Qed.

Lemma apply_eqb_nil (ap ap' : apply_expr) :
  apply_eqb [] [] ap ap' = true -> ap = ap'.
Proof.
  destruct ap, ap'. unfold apply_eqb, exn_extra_simples. simpl. intros Heq.
  tests_to_equations.
  match goal with
  | Hfst : map fst ?l = map fst ?l', Hsnd : map snd ?l = map snd ?l' |- _ =>
    pose proof (pairs_eq l l' Hfst Hsnd)
  end.
  congruence.
Qed.

Lemma arms_eqb_nil (arms arms' : list (Z * apply_cont)) :
  forall2b
    (fun arm arm' =>
       Z.eqb (fst arm) (fst arm')
       && apply_cont_eqb [] [] (snd arm) (snd arm'))
    arms arms'
  = true ->
  arms = arms'.
Proof.
  apply forall2b_eq. intros [i ac] [i' ac'] Heq. simpl in Heq.
  apply andb_true_iff in Heq as [Hi Hac]. apply Z.eqb_eq in Hi.
  apply apply_cont_eqb_nil in Hac. congruence.
Qed.

(* ======================================================================== *)
(* Maps that pair a name with itself only                                   *)
(* ======================================================================== *)

Definition diagonal (m : name_map) : Prop :=
  ∀ a a' : positive, names_match m a a' = true -> a = a'.

Lemma diagonal_nil : diagonal [].
Proof. exact names_match_nil. Qed.

Lemma diagonal_cons (m : name_map) (b : positive) :
  diagonal m -> diagonal ((b, b) :: m).
Proof.
  intros Hm a a'. simpl. destruct (Pos.eqb_spec a b) as [-> | _].
  - intros Heq. apply Pos.eqb_eq in Heq. congruence.
  - destruct (Pos.eqb a' b); [ discriminate | apply Hm ].
Qed.

Lemma bind_names_diagonal (xs : list positive) :
  ∀ m : name_map,
    diagonal m -> ∃ m' : name_map, bind_names xs xs m = Some m' ∧ diagonal m'.
Proof.
  induction xs as [ | x xs IH]; intros m Hm; simpl; [ eauto | ].
  apply IH, diagonal_cons, Hm.
Qed.

Lemma combine_diagonal (xs : list positive) (m : name_map) :
  diagonal m -> diagonal (combine xs xs ++ m).
Proof.
  intros Hm. induction xs as [ | x xs IH]; simpl; [ exact Hm | ].
  apply diagonal_cons, IH.
Qed.

Lemma composes_diagonal (m : name_map) : diagonal m -> composes m [] [].
Proof.
  intros Hm a a' a'' Hmatch Heq. apply Hm in Hmatch. subst a'. exact Heq.
Qed.

(* What is accepted under such maps is accepted with no names paired. *)
Lemma expr_eqb_diagonal (vm km : name_map) (e e' : expr) :
  diagonal vm -> diagonal km ->
  expr_eqb vm km e e' = true -> expr_eqb [] [] e e' = true.
Proof.
  intros Hvm Hkm Heq.
  exact (expr_eqb_trans e vm [] [] km [] [] (composes_diagonal vm Hvm)
           (composes_diagonal km Hkm) e' Heq e' (expr_eqb_reflexive e')).
Qed.

Lemma handler_eqb_diagonal (vm km : name_map) (h h' : continuation_handler) :
  diagonal vm -> diagonal km ->
  handler_eqb vm km h h' = true -> handler_eqb [] [] h h' = true.
Proof.
  intros Hvm Hkm Heq.
  exact (handler_eqb_trans h vm [] [] km [] [] (composes_diagonal vm Hvm)
           (composes_diagonal km Hkm) h' Heq h'
           (handler_eqb_refl h' [] [] reflexive_map_nil reflexive_map_nil)).
Qed.

Lemma bind_names_length (xs : list positive) :
  ∀ (xs' : list positive) (m m' : name_map),
    bind_names xs xs' m = Some m' -> length xs = length xs'.
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs'] m m'; simpl;
    try discriminate; [ reflexivity | ].
  intros Hbind. rewrite (IH xs' _ _ Hbind). reflexivity.
Qed.

Lemma paired_combine (f : positive -> positive) (xs : list positive) :
  paired f xs = combine xs (map f xs).
Proof.
  induction xs as [ | x xs IH]; simpl; [ reflexivity | ].
  rewrite <- IH. reflexivity.
Qed.

Lemma flip_combine_nil (xs xs' : list positive) :
  flip (combine xs xs' ++ []) = combine xs' xs ++ [].
Proof. rewrite !app_nil_r. apply flip_combine. Qed.

(* ======================================================================== *)
(* Opening a scope at fresh names                                           *)
(* ======================================================================== *)

(* The names [xs] that a binder binds may be exchanged for the names
   [zs], [free] being the free names of the scope: the bound names are
   different, the new names are as many, different, and neither bound
   nor free. *)
Definition opens_at (xs zs free : list positive) : Prop :=
  distinct xs = true
  ∧ length zs = length xs
  ∧ distinct_outside (xs ++ free) zs.

(* The renaming that exchanges the bound variables and the bound
   continuations of a scope for new ones. *)
Definition opening (xs zs : list variable) (ks zks : list continuation)
    : renaming :=
  Renaming (combine xs zs) (combine ks zks).

Lemma opens_at_nil (free : list positive) : opens_at [] [] free.
Proof.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | intros name Hname; discriminate ].
Qed.

(* Names that will do for two scopes at once. *)
Lemma fresh_opens_at (xs xs' free free' : list positive) :
  distinct xs = true -> distinct xs' = true -> length xs = length xs' ->
  ∃ zs : list positive, opens_at xs zs free ∧ opens_at xs' zs free'.
Proof.
  intros Hdistinct Hdistinct' Hlength.
  exists (fresh_names (length xs) ((xs ++ free) ++ xs' ++ free')).
  pose proof
    (fresh_names_outside (length xs) ((xs ++ free) ++ xs' ++ free'))
    as Houtside.
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  split; (split; [ assumption | split; [ | assumption ] ]).
  - apply fresh_names_length.
  - rewrite fresh_names_length. exact Hlength.
Qed.

Lemma opens_at_outside (xs zs free : list positive) (name : positive) :
  opens_at xs zs free -> among name zs = true ->
  among name xs = false ∧ among name free = false.
Proof.
  intros (_ & _ & _ & Houtside) Hname.
  apply among_app_false, Houtside, Hname.
Qed.

Lemma opens_at_moved (xs zs free : list positive) :
  opens_at xs zs free -> map (permute (combine xs zs)) xs = zs.
Proof.
  intros (Hdistinct & Hlength & Houtside).
  apply distinct_outside_app in Houtside as [Houtside _].
  exact (permute_combine xs zs (eq_sym Hlength) Hdistinct Houtside).
Qed.

Lemma opens_at_fixed (xs zs free : list positive) (name : positive) :
  opens_at xs zs free -> among name free = true -> name ∉ xs ->
  permute (combine xs zs) name = name.
Proof.
  intros Hopens Hfree Hnot_bound. apply permute_combine_other.
  - destruct (among name xs) eqn:Hamong; [ | reflexivity ].
    apply among_In in Hamong. contradiction.
  - destruct (among name zs) eqn:Hamong; [ | reflexivity ].
    apply (opens_at_outside xs zs free name Hopens) in Hamong as [_ Hamong].
    congruence.
Qed.

Section Opened.

Variables (xs zs : list variable) (ks zks : list continuation) (e : expr).

Hypothesis Hclosed : code_closed e = true.

Hypothesis Hxs : opens_at xs zs (free_variables_of_expr e).

Hypothesis Hks : opens_at ks zks (free_continuations_of_expr e).

Lemma opening_variables :
  map (rename_variable (opening xs zs ks zks)) xs = zs.
Proof. exact (opens_at_moved xs zs _ Hxs). Qed.

Lemma opening_continuations :
  map (rename_continuation (opening xs zs ks zks)) ks = zks.
Proof. exact (opens_at_moved ks zks _ Hks). Qed.

Lemma opening_fixes_variable (y : variable) :
  var_free_in_expr y e -> y ∉ xs ->
  rename_variable (opening xs zs ks zks) y = y.
Proof.
  intros Hfree. apply (opens_at_fixed xs zs _ y Hxs).
  apply (var_free_agrees y e Hclosed), Hfree.
Qed.

Lemma opening_fixes_continuation (k : continuation) :
  cont_free_in_expr k e -> k ∉ ks ->
  rename_continuation (opening xs zs ks zks) k = k.
Proof.
  intros Hfree. apply (opens_at_fixed ks zks _ k Hks).
  apply (cont_free_agrees k e Hclosed), Hfree.
Qed.

(* The comparison accepts the scope against the scope opened, under the
   maps that pair each bound name with its new name. *)
Lemma opening_eqb :
  ∃ vm : name_map,
    bind_names xs zs [] = Some vm
    ∧ expr_eqb vm (combine ks zks ++ []) e
        (rename_expr (opening xs zs ks zks) e)
      = true.
Proof.
  set (r := opening xs zs ks zks).
  assert (Hvariables : map (rename_variable r) xs = zs)
    by exact opening_variables.
  assert (Hcontinuations : map (rename_continuation r) ks = zks)
    by exact opening_continuations.
  exists (paired (rename_variable r) (rev xs) ++ []). split.
  - rewrite <- Hvariables at 1. apply bind_names_paired.
  - replace (combine ks zks) with (paired (rename_continuation r) ks)
      by (rewrite paired_combine;
          exact (f_equal (combine ks) Hcontinuations)).
    apply rename_expr_eqb; [ exact Hclosed | | ].
    + intros y Hy.
      apply (names_match_paired (rename_variable r) (rename_variable_eqb r)).
      destruct (in_dec Pos.eq_dec y xs) as [Hin | Hnot];
        [ left; apply in_rev in Hin; exact Hin | right ].
      assert (Hfixed : rename_variable r y = y)
        by exact (opening_fixes_variable y Hy Hnot).
      rewrite Hfixed. apply Pos.eqb_refl.
    + intros k Hk.
      apply (names_match_paired (rename_continuation r)
               (rename_continuation_eqb r)).
      destruct (in_dec Pos.eq_dec k ks) as [Hin | Hnot];
        [ left; exact Hin | right ].
      assert (Hfixed : rename_continuation r k = k)
        by exact (opening_fixes_continuation k Hk Hnot).
      rewrite Hfixed. apply Pos.eqb_refl.
Qed.

End Opened.

(* Two scopes that the comparison accepts under the pairs of their bound
   names are accepted with no names paired, once the bound names of
   both are exchanged for the same new names. *)
Lemma opened_eqb (xs xs' zs : list variable) (ks ks' zks : list continuation)
    (e e' : expr) (vm : name_map) :
  code_closed e = true ->
  code_closed e' = true ->
  opens_at xs zs (free_variables_of_expr e) ->
  opens_at xs' zs (free_variables_of_expr e') ->
  opens_at ks zks (free_continuations_of_expr e) ->
  opens_at ks' zks (free_continuations_of_expr e') ->
  bind_names xs xs' [] = Some vm ->
  expr_eqb vm (combine ks ks') e e' = true ->
  expr_eqb [] [] (rename_expr (opening xs zs ks zks) e)
    (rename_expr (opening xs' zs ks' zks) e')
  = true.
Proof.
  intros Hclosed Hclosed' Hxs Hxs' Hks Hks' Hbind Heq.
  destruct (opening_eqb xs zs ks zks e Hclosed Hxs Hks)
    as (vm1 & Hbind1 & Heq1).
  destruct (opening_eqb xs' zs ks' zks e' Hclosed' Hxs' Hks')
    as (vm2 & Hbind2 & Heq2).
  apply expr_eqb_flip in Heq1. rewrite flip_combine_nil in Heq1.
  rewrite <- (app_nil_r (combine ks ks')) in Heq.
  (* The pairs of variables compose to pairs of each new name with
     itself. *)
  assert (Hflip : bind_names zs xs [] = Some (flip vm1)).
  { pose proof (bind_names_flip xs zs []) as Hflip.
    rewrite Hbind1 in Hflip. exact Hflip. }
  destruct
    (bind_names_composes zs xs xs' [] [] [] (flip vm1) vm composes_nil Hflip
       Hbind)
    as (vm_middle & Hbind_middle & Hcomposes).
  destruct
    (bind_names_composes zs xs' zs [] [] [] vm_middle vm2 composes_nil
       Hbind_middle Hbind2)
    as (vm_last & Hbind_last & Hcomposes').
  destruct (bind_names_diagonal zs [] diagonal_nil)
    as (vm_diagonal & Hbind_diagonal & Hdiagonal).
  assert (Hsame : vm_last = vm_diagonal) by congruence. subst vm_diagonal.
  (* And so do the pairs of continuations. *)
  destruct Hks as (_ & Hlength & _). destruct Hks' as (_ & Hlength' & _).
  pose proof
    (combine_composes zks ks ks' [] [] [] Hlength
       (eq_trans (eq_sym Hlength) Hlength') composes_nil)
    as Hconts.
  pose proof
    (combine_composes zks ks' zks [] [] [] Hlength' (eq_sym Hlength')
       composes_nil)
    as Hconts'.
  pose proof (expr_eqb_trans _ _ _ _ _ _ _ Hcomposes Hconts _ Heq1 _ Heq)
    as Hmiddle.
  pose proof
    (expr_eqb_trans _ _ _ _ _ _ _ Hcomposes' Hconts' _ Hmiddle _ Heq2)
    as Hlast.
  exact (expr_eqb_diagonal _ _ _ _ Hdiagonal
           (combine_diagonal zks [] diagonal_nil) Hlast).
Qed.

(* ======================================================================== *)
(* What the recursion proves                                                *)
(* ======================================================================== *)

(* Of an expression, under any renaming. *)
Definition expr_converse (e : expr) : Prop :=
  ∀ (r : renaming) (e' : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    code_closed e' = true ->
    binders_distinct e' = true ->
    expr_eqb [] [] (rename_expr r e) e' = true ->
    alpha_equivalent (rename_expr r e) e'.

(* Of a handler: wherever it stands, in a non-recursive [Let_cont] or
   in a recursive one, it may be replaced by a handler that the
   comparison accepts against it. *)
Definition handler_converse (h : continuation_handler) : Prop :=
  ∀ (r : renaming) (h' : continuation_handler),
    handler_code_closed h = true ->
    handler_binders_distinct h = true ->
    handler_code_closed h' = true ->
    handler_binders_distinct h' = true ->
    handler_eqb [] [] (rename_handler r h) h' = true ->
    (∀ (k : continuation) (body : expr),
      alpha_equivalent (Let_cont_non_recursive k (rename_handler r h) body)
        (Let_cont_non_recursive k h' body))
    ∧ (∀ (before : list (continuation * continuation_handler))
         (k : continuation)
         (after : list (continuation * continuation_handler)) (body : expr),
        alpha_equivalent
          (Let_cont_recursive (before ++ (k, rename_handler r h) :: after)
             body)
          (Let_cont_recursive (before ++ (k, h') :: after) body)).

(* Of code: wherever it is bound. *)
Definition code_converse (c : code) : Prop :=
  ∀ (r : renaming) (c' : code),
    code_is_closed c = true ->
    code_binders_distinct c = true ->
    code_is_closed c' = true ->
    code_binders_distinct c' = true ->
    code_eqb (rename_code r c) c' = true ->
    ∀ (before : list static_binding) (id : code_id)
      (after : list static_binding) (body : expr),
      alpha_equivalent
        (Let_static (before ++ Bind_code id (rename_code r c) :: after) body)
        (Let_static (before ++ Bind_code id c' :: after) body).

(* ======================================================================== *)
(* What has no binder                                                       *)
(* ======================================================================== *)

Lemma apply_converse (ap : apply_expr) : expr_converse (Apply ap).
Proof.
  intros r e' _ _ _ _ Heq. cbn [rename_expr] in *.
  destruct e'; cbn [expr_eqb] in Heq; try discriminate Heq.
  apply apply_eqb_nil in Heq. rewrite Heq. apply Alpha_refl.
Qed.

Lemma apply_cont_converse (ac : apply_cont) : expr_converse (Apply_cont ac).
Proof.
  intros r e' _ _ _ _ Heq. cbn [rename_expr] in *.
  destruct e'; cbn [expr_eqb] in Heq; try discriminate Heq.
  apply apply_cont_eqb_nil in Heq. rewrite Heq. apply Alpha_refl.
Qed.

Lemma switch_converse (scrutinee : simple) (arms : list (Z * apply_cont)) :
  expr_converse (Switch scrutinee arms).
Proof.
  intros r e' _ _ _ _ Heq. cbn [rename_expr] in *.
  destruct e'; cbn [expr_eqb] in Heq; try discriminate Heq.
  apply andb_true_iff in Heq as [Hscrutinee Harms].
  apply simple_eqb_nil in Hscrutinee. apply arms_eqb_nil in Harms.
  rewrite Hscrutinee, Harms. apply Alpha_refl.
Qed.

Lemma invalid_converse : expr_converse Invalid.
Proof.
  intros r e' _ _ _ _ Heq. cbn [rename_expr] in *.
  destruct e'; cbn [expr_eqb] in Heq; try discriminate Heq.
  apply Alpha_refl.
Qed.

(* ======================================================================== *)
(* The variable of a [Let]                                                  *)
(* ======================================================================== *)

Lemma among_single (name : positive) : among name [name] = true.
Proof. rewrite among_cons, Pos.eqb_refl. reflexivity. Qed.

(* One new name. *)
Lemma opens_at_one (x : positive) (zs free : list positive) :
  opens_at [x] zs free ->
  ∃ z : positive, zs = [z] ∧ among z free = false.
Proof.
  intros Hopens. pose proof Hopens as (_ & Hlength & _).
  destruct zs as [ | z [ | other zs]]; try discriminate Hlength.
  exists z. split; [ reflexivity | ].
  exact (proj2 (opens_at_outside [x] [z] free z Hopens (among_single z))).
Qed.

Lemma let_converse (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  expr_converse body -> expr_converse (Let x kind d body).
Proof.
  intros IH r e' Hclosed Hdistinct Hclosed' Hdistinct' Heq.
  cbn [rename_expr] in *.
  destruct e' as [x' kind' d' body' | | | | | | | ]; cbn [expr_eqb] in Heq;
    try discriminate Heq.
  apply andb_true_iff in Heq as [Hd Hbody].
  apply andb_true_iff in Hd as [Hkind Hd]. apply kind_eqb_eq in Hkind.
  apply named_eqb_nil in Hd. subst kind' d'.
  cbn [code_closed binders_distinct] in Hclosed, Hdistinct, Hclosed',
    Hdistinct'.
  assert (Hclosed_renamed : code_closed (rename_expr r body) = true)
    by (rewrite code_closed_equivariant; exact Hclosed).
  destruct
    (fresh_opens_at [rename_variable r x] [x']
       (free_variables_of_expr (rename_expr r body))
       (free_variables_of_expr body') eq_refl eq_refl eq_refl)
    as (zs & Hopens & Hopens').
  destruct (opens_at_one _ _ _ Hopens) as (z & -> & Hfresh).
  destruct (opens_at_one _ _ _ Hopens') as (z' & Hsame & Hfresh').
  injection Hsame as <-.
  apply (alpha_let_at (rename_variable r x) x' z); try assumption.
  change (swap_variable (rename_variable r x) z)
    with (opening [rename_variable r x] [z] [] []).
  change (swap_variable x' z) with (opening [x'] [z] [] []).
  rewrite <- rename_expr_compose.
  apply IH; try assumption.
  - rewrite code_closed_equivariant. exact Hclosed'.
  - rewrite binders_distinct_equivariant. exact Hdistinct'.
  - rewrite rename_expr_compose.
    apply (opened_eqb [rename_variable r x] [x'] [z] [] [] [] _ _
             [(rename_variable r x, x')]);
      solve [ assumption | apply opens_at_nil | reflexivity ].
Qed.

(* ======================================================================== *)
(* A handler and its parameters                                             *)
(* ======================================================================== *)

(* FlambdaFreshBindersSpec.alpha_handler_at, for a handler of a
   recursive [Let_cont]. *)
Lemma alpha_recursive_handler_swap
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params fresh : list variable) (metadata : handler_metadata)
    (handler : expr) (after : list (continuation * continuation_handler))
    (body : expr) :
  code_closed handler = true ->
  distinct params = true ->
  length fresh = length params ->
  distinct_outside (params ++ free_variables_of_expr handler) fresh ->
  alpha_equivalent
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
    (Let_cont_recursive
       (before
        ++ (k,
            Handler fresh metadata
              (rename_expr (swap_variables params fresh) handler))
        :: after)
       body).
Proof.
  intros Hclosed Hdistinct Hlength Houtside.
  destruct (swap_variables_fixes params fresh handler Hclosed Houtside)
    as [Hvariables Hcontinuations].
  apply distinct_outside_app in Houtside as [Hparams _].
  pose proof
    (Alpha_recursive_handler_by_relation (swap_variables params fresh) before
       k params metadata handler after body Hclosed Hvariables
       Hcontinuations)
    as Hrule.
  cbn [rename_handler] in Hrule.
  rewrite (rename_variables_swap params fresh Hlength Hdistinct Hparams)
    in Hrule.
  exact Hrule.
Qed.

Lemma alpha_recursive_handler_at
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params params' fresh : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr) :
  code_closed handler = true ->
  code_closed handler' = true ->
  distinct params = true ->
  distinct params' = true ->
  length fresh = length params ->
  length fresh = length params' ->
  distinct_outside (params ++ free_variables_of_expr handler) fresh ->
  distinct_outside (params' ++ free_variables_of_expr handler') fresh ->
  alpha_equivalent (rename_expr (swap_variables params fresh) handler)
    (rename_expr (swap_variables params' fresh) handler') ->
  alpha_equivalent
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
    (Let_cont_recursive
       (before ++ (k, Handler params' metadata handler') :: after) body).
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Hlength Hlength' Houtside
    Houtside' Hhandlers.
  apply Alpha_trans
    with (Let_cont_recursive
            (before
             ++ (k,
                 Handler fresh metadata
                   (rename_expr (swap_variables params fresh) handler))
             :: after)
            body);
    [ apply alpha_recursive_handler_swap; assumption | ].
  apply Alpha_trans
    with (Let_cont_recursive
            (before
             ++ (k,
                 Handler fresh metadata
                   (rename_expr (swap_variables params' fresh) handler'))
             :: after)
            body);
    [ | apply Alpha_sym, alpha_recursive_handler_swap; assumption ].
  exact
    (Alpha_context
       (Let_cont_recursive_handler before k fresh metadata Hole after body) _
       _ Hhandlers).
Qed.

Lemma handler_converse_intro (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  expr_converse handler -> handler_converse (Handler params metadata handler).
Proof.
  intros IH r [params' metadata' handler'] Hclosed Hdistinct Hclosed'
    Hdistinct' Heq.
  pose proof
    (handler_binders_distinct_renamed r (Handler params metadata handler))
    as Hdistinct_renamed.
  rewrite Hdistinct in Hdistinct_renamed.
  cbn [rename_handler handler_eqb handler_code_closed
       handler_binders_distinct] in *.
  destruct (bind_names (map (rename_variable r) params) params' [])
    as [vm | ] eqn:Hbind; [ | discriminate Heq ].
  apply andb_true_iff in Heq as [Hmetadata Heq].
  apply handler_metadata_eqb_eq in Hmetadata. subst metadata'.
  apply andb_true_iff in Hdistinct as [_ Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hparams' Hdistinct'].
  apply andb_true_iff in Hdistinct_renamed as [Hparams _].
  assert (Hclosed_renamed : code_closed (rename_expr r handler) = true)
    by (rewrite code_closed_equivariant; exact Hclosed).
  destruct
    (fresh_opens_at (map (rename_variable r) params) params'
       (free_variables_of_expr (rename_expr r handler))
       (free_variables_of_expr handler') Hparams Hparams'
       (bind_names_length _ _ _ _ Hbind))
    as (zs & Hopens & Hopens').
  assert (Hscopes :
    alpha_equivalent
      (rename_expr (swap_variables (map (rename_variable r) params) zs)
         (rename_expr r handler))
      (rename_expr (swap_variables params' zs) handler')).
  { change (swap_variables (map (rename_variable r) params) zs)
      with (opening (map (rename_variable r) params) zs [] []).
    change (swap_variables params' zs) with (opening params' zs [] []).
    rewrite <- rename_expr_compose.
    apply IH; try assumption.
    - rewrite code_closed_equivariant. exact Hclosed'.
    - rewrite binders_distinct_equivariant. exact Hdistinct'.
    - rewrite rename_expr_compose.
      apply (opened_eqb (map (rename_variable r) params) params' zs [] [] []
               _ _ vm);
        solve [ assumption | apply opens_at_nil ]. }
  destruct Hopens as (_ & Hlength & Houtside).
  destruct Hopens' as (_ & Hlength' & Houtside').
  split.
  - intros k body. apply (alpha_handler_at k _ params' zs); assumption.
  - intros before k after body.
    apply (alpha_recursive_handler_at before k _ params' zs); assumption.
Qed.

(* ======================================================================== *)
(* The continuation of a non-recursive [Let_cont]                           *)
(* ======================================================================== *)

Lemma let_cont_converse (k : continuation) (h : continuation_handler)
    (body : expr) :
  handler_converse h -> expr_converse body ->
  expr_converse (Let_cont_non_recursive k h body).
Proof.
  intros IHh IH r e' Hclosed Hdistinct Hclosed' Hdistinct' Heq.
  cbn [rename_expr] in *.
  destruct e' as [ | | k' h' body' | | | | | ]; cbn [expr_eqb] in Heq;
    try discriminate Heq.
  apply andb_true_iff in Heq as [Hh Hbody].
  cbn [code_closed binders_distinct] in Hclosed, Hdistinct, Hclosed',
    Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_h Hclosed].
  apply andb_true_iff in Hdistinct as [Hdistinct_h Hdistinct].
  apply andb_true_iff in Hclosed' as [Hclosed_h' Hclosed'].
  apply andb_true_iff in Hdistinct' as [Hdistinct_h' Hdistinct'].
  apply Alpha_trans
    with (Let_cont_non_recursive k' (rename_handler r h) body');
    [ | apply (IHh r h'); assumption ].
  assert (Hclosed_renamed : code_closed (rename_expr r body) = true)
    by (rewrite code_closed_equivariant; exact Hclosed).
  destruct
    (fresh_opens_at [rename_continuation r k] [k']
       (free_continuations_of_expr (rename_expr r body))
       (free_continuations_of_expr body') eq_refl eq_refl eq_refl)
    as (zs & Hopens & Hopens').
  destruct (opens_at_one _ _ _ Hopens) as (z & -> & Hfresh).
  destruct (opens_at_one _ _ _ Hopens') as (z' & Hsame & Hfresh').
  injection Hsame as <-.
  apply (alpha_let_cont_at (rename_continuation r k) k' z); try assumption.
  change (swap_continuation (rename_continuation r k) z)
    with (opening [] [] [rename_continuation r k] [z]).
  change (swap_continuation k' z) with (opening [] [] [k'] [z]).
  rewrite <- rename_expr_compose.
  apply IH; try assumption.
  - rewrite code_closed_equivariant. exact Hclosed'.
  - rewrite binders_distinct_equivariant. exact Hdistinct'.
  - rewrite rename_expr_compose.
    apply (opened_eqb [] [] [] [rename_continuation r k] [k'] [z] _ _ []);
      solve [ assumption | apply opens_at_nil | reflexivity ].
Qed.

(* ======================================================================== *)
(* Code and its binders                                                     *)
(* ======================================================================== *)

Lemma distinct_two (a b : positive) :
  negb (Pos.eqb a b) = true -> distinct [a; b] = true.
Proof.
  intros Hdifferent. apply negb_true_iff in Hdifferent. cbn [distinct].
  rewrite among_cons, Hdifferent. reflexivity.
Qed.

(* The binders of closed code exchanged for new names. *)
Lemma alpha_code_swap (before : list static_binding) (id : code_id)
    (params : list variable) (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (function_body : expr)
    (new_my_closure : variable)
    (new_params : list variable) (new_ret new_exn : continuation)
    (after : list static_binding) (body : expr) :
  code_is_closed (Code params my_closure ret exn metadata function_body)
  = true ->
  opens_at (my_closure :: params) (new_my_closure :: new_params)
    (free_variables_of_expr function_body) ->
  opens_at [ret; exn] [new_ret; new_exn]
    (free_continuations_of_expr function_body) ->
  alpha_equivalent
    (Let_static
       (before
        ++ Bind_code id
             (Code params my_closure ret exn metadata function_body)
        :: after)
       body)
    (Let_static
       (before
        ++ Bind_code id
             (Code new_params new_my_closure new_ret new_exn metadata
                (rename_expr
                   (opening (my_closure :: params)
                      (new_my_closure :: new_params) [ret; exn]
                      [new_ret; new_exn])
                   function_body))
        :: after)
       body).
Proof.
  intros Hclosed Hxs Hks.
  set (r :=
    opening (my_closure :: params) (new_my_closure :: new_params) [ret; exn]
      [new_ret; new_exn]).
  pose proof
    (Alpha_code r before id
       (Code params my_closure ret exn metadata function_body) after body
       Hclosed)
    as Hrule.
  cbn [rename_code] in Hrule.
  pose proof (opens_at_moved _ _ _ Hxs) as Hvariables.
  cbn [map] in Hvariables. injection Hvariables as Hmy_closure Hparams.
  pose proof (opens_at_moved _ _ _ Hks) as Hcontinuations.
  cbn [map] in Hcontinuations. injection Hcontinuations as Hret Hexn.
  replace (rename_variable r my_closure) with new_my_closure in Hrule
    by (symmetry; exact Hmy_closure).
  replace (map (rename_variable r) params) with new_params in Hrule
    by (symmetry; exact Hparams).
  replace (rename_continuation r ret) with new_ret in Hrule
    by (symmetry; exact Hret).
  replace (rename_continuation r exn) with new_exn in Hrule
    by (symmetry; exact Hexn).
  exact Hrule.
Qed.

Lemma code_converse_intro (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (function_body : expr) :
  expr_converse function_body ->
  code_converse (Code params my_closure ret exn metadata function_body).
Proof.
  intros IH r [params' my_closure' ret' exn' metadata' function_body'] Hclosed
    Hdistinct Hclosed' Hdistinct' Heq before id after body.
  pose proof
    (code_is_closed_renamed r
       (Code params my_closure ret exn metadata function_body))
    as Hclosed_renamed.
  rewrite Hclosed in Hclosed_renamed.
  pose proof
    (code_binders_distinct_renamed r
       (Code params my_closure ret exn metadata function_body))
    as Hdistinct_renamed.
  rewrite Hdistinct in Hdistinct_renamed.
  cbn [rename_code] in *.
  set (params1 := map (rename_variable r) params) in *.
  set (my_closure1 := rename_variable r my_closure) in *.
  set (ret1 := rename_continuation r ret) in *.
  set (exn1 := rename_continuation r exn) in *.
  cbn [code_eqb] in Heq.
  destruct (bind_names params1 params' [(my_closure1, my_closure')])
    as [vm | ] eqn:Hbind; [ | discriminate Heq ].
  apply andb_true_iff in Heq as [Hmetadata Heq].
  apply code_metadata_eqb_eq in Hmetadata. subst metadata'.
  (* Closed code, said of the three bodies. *)
  destruct (proj1 (closed_code _) Hclosed) as [_ Hclosed_body].
  destruct (proj1 (closed_code _) Hclosed_renamed) as [_ Hclosed_body1].
  destruct (proj1 (closed_code _) Hclosed') as [_ Hclosed_body'].
  cbn [code_body] in Hclosed_body, Hclosed_body1, Hclosed_body'.
  (* The binders are different names. *)
  cbn [code_binders_distinct] in Hdistinct, Hdistinct', Hdistinct_renamed.
  apply andb_true_iff in Hdistinct as [_ Hdistinct_body].
  apply andb_true_iff in Hdistinct' as [Hbinders' Hdistinct_body'].
  apply andb_true_iff in Hbinders' as [Hvariables' Hcontinuations'].
  apply andb_true_iff in Hdistinct_renamed as [Hbinders _].
  apply andb_true_iff in Hbinders as [Hvariables Hcontinuations].
  assert (Hlength :
    length (my_closure1 :: params1) = length (my_closure' :: params'))
    by exact (f_equal S (bind_names_length _ _ _ _ Hbind)).
  destruct
    (fresh_opens_at (my_closure1 :: params1) (my_closure' :: params')
       (free_variables_of_expr (rename_expr r function_body))
       (free_variables_of_expr function_body') Hvariables Hvariables' Hlength)
    as (zs & Hxs & Hxs').
  destruct
    (fresh_opens_at [ret1; exn1] [ret'; exn']
       (free_continuations_of_expr (rename_expr r function_body))
       (free_continuations_of_expr function_body')
       (distinct_two _ _ Hcontinuations) (distinct_two _ _ Hcontinuations')
       eq_refl)
    as (zks & Hks & Hks').
  destruct zs as [ | new_my_closure new_params];
    [ destruct Hxs as (_ & Hl & _); discriminate Hl | ].
  destruct zks as [ | new_ret [ | new_exn [ | other zks]]];
    try (destruct Hks as (_ & Hl & _); discriminate Hl).
  assert (Hscopes :
    alpha_equivalent
      (rename_expr
         (opening (my_closure1 :: params1) (new_my_closure :: new_params)
            [ret1; exn1] [new_ret; new_exn])
         (rename_expr r function_body))
      (rename_expr
         (opening (my_closure' :: params') (new_my_closure :: new_params)
            [ret'; exn'] [new_ret; new_exn])
         function_body')).
  { rewrite <- rename_expr_compose.
    apply IH; try assumption.
    - rewrite code_closed_equivariant. exact Hclosed_body'.
    - rewrite binders_distinct_equivariant. exact Hdistinct_body'.
    - rewrite rename_expr_compose.
      apply (opened_eqb (my_closure1 :: params1) (my_closure' :: params')
               (new_my_closure :: new_params) [ret1; exn1] [ret'; exn']
               [new_ret; new_exn] _ _ vm);
        assumption. }
  apply Alpha_trans
    with (Let_static
            (before
             ++ Bind_code id
                  (Code new_params new_my_closure new_ret new_exn metadata
                     (rename_expr
                        (opening (my_closure1 :: params1)
                           (new_my_closure :: new_params) [ret1; exn1]
                           [new_ret; new_exn])
                        (rename_expr r function_body)))
             :: after)
            body);
    [ apply alpha_code_swap; assumption | ].
  apply Alpha_trans
    with (Let_static
            (before
             ++ Bind_code id
                  (Code new_params new_my_closure new_ret new_exn metadata
                     (rename_expr
                        (opening (my_closure' :: params')
                           (new_my_closure :: new_params) [ret'; exn']
                           [new_ret; new_exn])
                        function_body'))
             :: after)
            body);
    [ | apply Alpha_sym, alpha_code_swap; assumption ].
  exact
    (Alpha_context
       (Let_static_code before id new_params new_my_closure new_ret new_exn
          metadata Hole after body)
       _ _ Hscopes).
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Definition binding_converse (binding : static_binding) : Prop :=
  match binding with
  | Bind_code _ c => code_converse c
  | Bind_deleted_code _ | Bind_symbol _ _ => True
  end.

(* One binding after another: [done] is what has been replaced. *)
Lemma bindings_converse (r : renaming) (body : expr)
    (bindings : list static_binding) :
  Forall binding_converse bindings ->
  ∀ done bindings' : list static_binding,
    forallb binding_code_closed bindings = true ->
    forallb binding_binders_distinct bindings = true ->
    forallb binding_code_closed bindings' = true ->
    forallb binding_binders_distinct bindings' = true ->
    forall2b (binding_eqb []) (map (rename_static_binding r) bindings)
      bindings'
    = true ->
    alpha_equivalent
      (Let_static (done ++ map (rename_static_binding r) bindings) body)
      (Let_static (done ++ bindings') body).
Proof.
  intros Hall.
  induction Hall as [ | binding bindings Hbinding _ IH];
    intros done [ | binding' bindings'] Hclosed Hdistinct Hclosed' Hdistinct'
      Heq;
    cbn [map forall2b] in Heq; try discriminate Heq.
  - apply Alpha_refl.
  - apply andb_true_iff in Heq as [Hfirst Hrest].
    cbn [forallb] in Hclosed, Hdistinct, Hclosed', Hdistinct'.
    apply andb_true_iff in Hclosed as [Hclosed_first Hclosed].
    apply andb_true_iff in Hdistinct as [Hdistinct_first Hdistinct].
    apply andb_true_iff in Hclosed' as [Hclosed_first' Hclosed'].
    apply andb_true_iff in Hdistinct' as [Hdistinct_first' Hdistinct'].
    cbn [map].
    apply Alpha_trans
      with (Let_static
              (done ++ binding' :: map (rename_static_binding r) bindings)
              body).
    + destruct binding as [id c | id | s const],
        binding' as [id' c' | id' | s' const'];
        cbn [rename_static_binding binding_eqb] in Hfirst |- *;
        try discriminate Hfirst.
      * apply andb_true_iff in Hfirst as [Hid Hcode].
        apply Pos.eqb_eq in Hid. subst id'.
        apply (Hbinding r c'); assumption.
      * apply Pos.eqb_eq in Hfirst. subst id'. apply Alpha_refl.
      * apply andb_true_iff in Hfirst as [Hs Hconst].
        apply Pos.eqb_eq in Hs. apply static_const_eqb_nil in Hconst.
        subst s'. rewrite Hconst. apply Alpha_refl.
    + replace (done ++ binding' :: map (rename_static_binding r) bindings)
        with ((done ++ [binding']) ++ map (rename_static_binding r) bindings)
        by (rewrite <- app_assoc; reflexivity).
      replace (done ++ binding' :: bindings')
        with ((done ++ [binding']) ++ bindings')
        by (rewrite <- app_assoc; reflexivity).
      apply IH; assumption.
Qed.

Lemma let_static_converse (bindings : list static_binding) (body : expr) :
  Forall binding_converse bindings -> expr_converse body ->
  expr_converse (Let_static bindings body).
Proof.
  intros Hbindings IH r e' Hclosed Hdistinct Hclosed' Hdistinct' Heq.
  cbn [rename_expr] in *.
  destruct e' as [ | bindings' body' | | | | | | ]; cbn [expr_eqb] in Heq;
    try discriminate Heq.
  apply andb_true_iff in Heq as [Hgroup Hbody].
  cbn [code_closed binders_distinct] in Hclosed, Hdistinct, Hclosed',
    Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_group Hclosed].
  apply andb_true_iff in Hdistinct as [Hdistinct_group Hdistinct].
  apply andb_true_iff in Hclosed' as [Hclosed_group' Hclosed'].
  apply andb_true_iff in Hdistinct' as [Hdistinct_group' Hdistinct'].
  apply Alpha_trans
    with (Let_static (map (rename_static_binding r) bindings) body').
  - exact
      (Alpha_context
         (Let_static_body (map (rename_static_binding r) bindings) Hole) _ _
         (IH r body' Hclosed Hdistinct Hclosed' Hdistinct' Hbody)).
  - exact
      (bindings_converse r body' bindings Hbindings [] bindings'
         Hclosed_group Hdistinct_group Hclosed_group' Hdistinct_group'
         Hgroup).
Qed.

(* ======================================================================== *)
(* The continuations of a recursive [Let_cont]                              *)
(* ======================================================================== *)

Lemma forall2b_impl {A B : Type} (f g : A -> B -> bool) :
  (∀ (a : A) (b : B), f a b = true -> g a b = true) ->
  ∀ (l : list A) (l' : list B),
    forall2b f l l' = true -> forall2b g l l' = true.
Proof.
  intros Hfg. induction l as [ | a l IH]; intros [ | b l']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hab Hl].
  rewrite (Hfg a b Hab), (IH l' Hl). reflexivity.
Qed.

Lemma combine_names_length (xs xs' : list positive) (m m' : name_map) :
  combine_names xs xs' m = Some m' -> length xs = length xs'.
Proof.
  unfold combine_names.
  destruct (Nat.eqb_spec (length xs) (length xs')) as [Hlength | _];
    [ intros _; exact Hlength | discriminate ].
Qed.

(* One handler after another: [done] is what has been replaced. The
   handlers of the two lists have the same continuations. *)
Lemma handlers_converse (r : renaming) (body : expr)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun binding => handler_converse (snd binding)) handlers ->
  ∀ done handlers' : list (continuation * continuation_handler),
    forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
    forallb (fun '(_, h) => handler_binders_distinct h) handlers = true ->
    forallb (fun '(_, h) => handler_code_closed h) handlers' = true ->
    forallb (fun '(_, h) => handler_binders_distinct h) handlers' = true ->
    map fst (rename_handlers r handlers) = map fst handlers' ->
    forall2b (fun '(_, h) '(_, h') => handler_eqb [] [] h h')
      (rename_handlers r handlers) handlers'
    = true ->
    alpha_equivalent
      (Let_cont_recursive (done ++ rename_handlers r handlers) body)
      (Let_cont_recursive (done ++ handlers') body).
Proof.
  intros Hall.
  induction Hall as [ | [k h] handlers Hhandler _ IH];
    intros done [ | [k' h'] handlers'] Hclosed Hdistinct Hclosed' Hdistinct'
      Hnames Heq;
    cbn [rename_handlers map forall2b] in Heq, Hnames |- *;
    try discriminate Heq.
  - apply Alpha_refl.
  - cbv beta iota in Heq. apply andb_true_iff in Heq as [Hfirst Hrest].
    cbn [fst] in Hnames. injection Hnames as Hname Hnames. subst k'.
    cbn [forallb] in Hclosed, Hdistinct, Hclosed', Hdistinct'.
    apply andb_true_iff in Hclosed as [Hclosed_first Hclosed].
    apply andb_true_iff in Hdistinct as [Hdistinct_first Hdistinct].
    apply andb_true_iff in Hclosed' as [Hclosed_first' Hclosed'].
    apply andb_true_iff in Hdistinct' as [Hdistinct_first' Hdistinct'].
    cbn [snd] in Hhandler.
    apply Alpha_trans
      with (Let_cont_recursive
              (done ++ (rename_continuation r k, h')
               :: rename_handlers r handlers)
              body).
    + exact
        (proj2
           (Hhandler r h' Hclosed_first Hdistinct_first Hclosed_first'
              Hdistinct_first' Hfirst)
           done (rename_continuation r k) (rename_handlers r handlers) body).
    + replace
        (done ++ (rename_continuation r k, h') :: rename_handlers r handlers)
        with
          ((done ++ [(rename_continuation r k, h')])
           ++ rename_handlers r handlers)
        by (rewrite <- app_assoc; reflexivity).
      replace (done ++ (rename_continuation r k, h') :: handlers')
        with ((done ++ [(rename_continuation r k, h')]) ++ handlers')
        by (rewrite <- app_assoc; reflexivity).
      apply IH; assumption.
Qed.

(* The swaps of the continuations of a group with new names fix the
   free names of the whole expression. *)
Lemma group_fixes (zks : list continuation)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  code_closed (Let_cont_recursive handlers body) = true ->
  opens_at (map fst handlers) zks
    (free_continuations_of_expr (Let_cont_recursive handlers body)) ->
  (∀ x : variable,
    var_free_in_expr x (Let_cont_recursive handlers body) ->
    rename_variable (opening [] [] (map fst handlers) zks) x = x)
  ∧ (∀ k : continuation,
      cont_free_in_expr k (Let_cont_recursive handlers body) ->
      rename_continuation (opening [] [] (map fst handlers) zks) k = k).
Proof.
  intros Hclosed Hopens. split; [ intros x _; reflexivity | ].
  intros k Hk. apply (cont_free_agrees k _ Hclosed) in Hk.
  apply (opens_at_fixed _ zks _ k Hopens Hk).
  cbn [free_continuations_of_expr] in Hk. rewrite among_without in Hk.
  apply andb_true_iff in Hk as [_ Hk]. apply negb_true_iff in Hk.
  intros Hin. apply among_In in Hin. congruence.
Qed.

Lemma let_cont_recursive_converse
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun binding => handler_converse (snd binding)) handlers ->
  expr_converse body ->
  expr_converse (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers IH r e' Hclosed Hdistinct Hclosed' Hdistinct' Heq.
  destruct e' as [ | | | handlers' body' | | | | ];
    try (cbn [rename_expr expr_eqb] in Heq; discriminate Heq).
  change (rename_expr r (Let_cont_recursive handlers body))
    with (Let_cont_recursive (rename_handlers r handlers)
            (rename_expr r body))
    in *.
  assert (Hclosed1 :
    code_closed
      (Let_cont_recursive (rename_handlers r handlers) (rename_expr r body))
    = true)
    by (exact (eq_trans
                 (code_closed_equivariant r
                    (Let_cont_recursive handlers body))
                 Hclosed)).
  assert (Hdistinct1 :
    binders_distinct
      (Let_cont_recursive (rename_handlers r handlers) (rename_expr r body))
    = true)
    by (exact (eq_trans
                 (binders_distinct_equivariant r
                    (Let_cont_recursive handlers body))
                 Hdistinct)).
  (* The continuations of each group are different names, and the two
     groups have as many. *)
  pose proof Hdistinct1 as Hnames. cbn [binders_distinct] in Hnames.
  apply andb_true_iff in Hnames as [Hnames _].
  apply andb_true_iff in Hnames as [Hnames _].
  apply andb_true_iff in Hnames as [_ Hnames].
  pose proof Hdistinct' as Hnames'. cbn [binders_distinct] in Hnames'.
  apply andb_true_iff in Hnames' as [Hnames' _].
  apply andb_true_iff in Hnames' as [Hnames' _].
  apply andb_true_iff in Hnames' as [_ Hnames'].
  assert (Hlength :
    length (map fst (rename_handlers r handlers))
    = length (map fst handlers')).
  { cbn [expr_eqb] in Heq.
    destruct
      (combine_names (map fst (rename_handlers r handlers))
         (map fst handlers') [])
      as [km | ] eqn:Hcombine; [ | discriminate Heq ].
    exact (combine_names_length _ _ _ _ Hcombine). }
  destruct
    (fresh_opens_at (map fst (rename_handlers r handlers)) (map fst handlers')
       (free_continuations_of_expr
          (Let_cont_recursive (rename_handlers r handlers)
             (rename_expr r body)))
       (free_continuations_of_expr (Let_cont_recursive handlers' body'))
       Hnames Hnames' Hlength)
    as (zks & Hopens & Hopens').
  set (s := opening [] [] (map fst (rename_handlers r handlers)) zks).
  set (s' := opening [] [] (map fst handlers') zks).
  destruct (group_fixes zks _ _ Hclosed1 Hopens)
    as [Hvariables Hcontinuations].
  destruct (group_fixes zks _ _ Hclosed' Hopens')
    as [Hvariables' Hcontinuations'].
  (* Exchange the continuations of both groups for the new names. *)
  apply Alpha_trans
    with (rename_expr s
            (Let_cont_recursive (rename_handlers r handlers)
               (rename_expr r body)));
    [ exact (Alpha_rename_by_relation s _ Hclosed1 Hvariables
               Hcontinuations)
    | ].
  apply Alpha_trans
    with (rename_expr s' (Let_cont_recursive handlers' body'));
    [ | exact (Alpha_sym _ _
                 (Alpha_rename_by_relation s' _ Hclosed' Hvariables'
                    Hcontinuations')) ].
  assert (Hopened :
    expr_eqb [] []
      (rename_expr s
         (Let_cont_recursive (rename_handlers r handlers)
            (rename_expr r body)))
      (rename_expr s' (Let_cont_recursive handlers' body'))
    = true).
  { apply expr_eqb_transitive with (Let_cont_recursive handlers' body');
      [ | exact (rename_expr_accepted s' _ Hclosed' Hvariables'
                   Hcontinuations') ].
    apply expr_eqb_transitive
      with (Let_cont_recursive (rename_handlers r handlers)
              (rename_expr r body));
      [ | exact Heq ].
    apply expr_eqb_symmetric.
    exact (rename_expr_accepted s _ Hclosed1 Hvariables Hcontinuations). }
  assert (Hclosed2 :
    code_closed (rename_expr s' (Let_cont_recursive handlers' body')) = true)
    by (rewrite code_closed_equivariant; exact Hclosed').
  assert (Hdistinct2 :
    binders_distinct (rename_expr s' (Let_cont_recursive handlers' body'))
    = true)
    by (rewrite binders_distinct_equivariant; exact Hdistinct').
  (* Both groups now bind the new names. *)
  assert (Hmoved : map fst (rename_handlers s (rename_handlers r handlers))
                   = zks)
    by (rewrite map_fst_rename_handlers; exact (opens_at_moved _ _ _ Hopens)).
  assert (Hmoved' : map fst (rename_handlers s' handlers') = zks)
    by (rewrite map_fst_rename_handlers;
        exact (opens_at_moved _ _ _ Hopens')).
  (* The first term is the term given under one renaming. *)
  assert (Hfirst :
    rename_expr s
      (Let_cont_recursive (rename_handlers r handlers) (rename_expr r body))
    = Let_cont_recursive (rename_handlers (compose_renaming s r) handlers)
        (rename_expr (compose_renaming s r) body))
    by (exact (eq_sym
                 (rename_expr_compose s r
                    (Let_cont_recursive handlers body)))).
  pose proof Hfirst as Hsame. cbn [rename_expr] in Hsame.
  injection Hsame as Hsame_handlers _.
  assert (Hmoved_first :
    map fst (rename_handlers (compose_renaming s r) handlers) = zks)
    by (rewrite <- Hsame_handlers; exact Hmoved).
  rewrite Hfirst in Hopened |- *.
  change (rename_expr s' (Let_cont_recursive handlers' body'))
    with (Let_cont_recursive (rename_handlers s' handlers')
            (rename_expr s' body'))
    in *.
  cbn [expr_eqb] in Hopened. rewrite Hmoved_first, Hmoved' in Hopened.
  unfold combine_names in Hopened. rewrite Nat.eqb_refl in Hopened.
  apply andb_true_iff in Hopened as [Hgroup Hbodies].
  apply (expr_eqb_diagonal [] (combine zks zks ++ []) _ _ diagonal_nil
           (combine_diagonal zks [] diagonal_nil))
    in Hbodies.
  apply
    (forall2b_impl _ (fun '(_, h) '(_, h') => handler_eqb [] [] h h'))
    in Hgroup;
    [ | intros [k h] [k' h'];
        exact (handler_eqb_diagonal [] (combine zks zks ++ []) h h'
                 diagonal_nil (combine_diagonal zks [] diagonal_nil)) ].
  (* What is known of the parts. *)
  cbn [code_closed] in Hclosed, Hclosed2.
  apply andb_true_iff in Hclosed as [Hclosed_handlers Hclosed_body].
  apply andb_true_iff in Hclosed2 as [Hclosed_handlers2 Hclosed_body2].
  cbn [binders_distinct] in Hdistinct, Hdistinct2.
  apply andb_true_iff in Hdistinct as [Hdistinct Hdistinct_body].
  apply andb_true_iff in Hdistinct as [_ Hdistinct_handlers].
  apply andb_true_iff in Hdistinct2 as [Hdistinct2 Hdistinct_body2].
  apply andb_true_iff in Hdistinct2 as [_ Hdistinct_handlers2].
  apply Alpha_trans
    with (Let_cont_recursive
            (rename_handlers (compose_renaming s r) handlers)
            (rename_expr s' body')).
  - exact
      (Alpha_context
         (Let_cont_recursive_body
            (rename_handlers (compose_renaming s r) handlers) Hole)
         _ _
         (IH (compose_renaming s r) (rename_expr s' body') Hclosed_body
            Hdistinct_body Hclosed_body2 Hdistinct_body2 Hbodies)).
  - exact
      (handlers_converse (compose_renaming s r) (rename_expr s' body')
         handlers IHhandlers [] (rename_handlers s' handlers')
         Hclosed_handlers Hdistinct_handlers Hclosed_handlers2
         Hdistinct_handlers2 (eq_trans Hmoved_first (eq_sym Hmoved'))
         Hgroup).
Qed.

(* ======================================================================== *)
(* The recursion                                                            *)
(* ======================================================================== *)

Fixpoint expr_converse_all (e : expr) : expr_converse e
with handler_converse_all (h : continuation_handler) : handler_converse h
with code_converse_all (c : code) : code_converse c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ].
    + apply let_converse, expr_converse_all.
    + apply let_static_converse; [ | apply expr_converse_all ].
      induction bindings as [ | binding bindings IH]; constructor;
        [ | exact IH ].
      destruct binding as [id c | id | s const];
        [ exact (code_converse_all c) | exact I | exact I ].
    + apply let_cont_converse;
        [ apply handler_converse_all | apply expr_converse_all ].
    + apply let_cont_recursive_converse; [ | apply expr_converse_all ].
      induction handlers as [ | [k h] handlers IH]; constructor;
        [ exact (handler_converse_all h) | exact IH ].
    + apply apply_converse.
    + apply apply_cont_converse.
    + apply switch_converse.
    + apply invalid_converse.
  - destruct h as [params metadata handler].
    apply handler_converse_intro, expr_converse_all.
  - destruct c as [params my_closure ret exn metadata body].
    apply code_converse_intro, expr_converse_all.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem expr_eqb_alpha_equivalent : expr_eqb_alpha_equivalent_statement.
Proof.
  intros e e' Hclosed Hclosed' Hdistinct Hdistinct' Heq.
  pose proof
    (expr_converse_all e identity_renaming e' Hclosed Hdistinct Hclosed'
       Hdistinct')
    as Hconverse.
  rewrite rename_expr_identity in Hconverse. exact (Hconverse Heq).
Qed.

Theorem expr_eqb_iff_alpha_equivalent
  : expr_eqb_iff_alpha_equivalent_statement.
Proof.
  intros e e' Hclosed Hclosed' Hdistinct Hdistinct'. split.
  - apply expr_eqb_alpha_equivalent; assumption.
  - apply alpha_equivalent_accepted.
Qed.

(* From left to right: the comparison accepts the two [Let]s, so it
   accepts the bodies under the pair of the bound variables, so it
   accepts them opened at a fresh name, and what it accepts is
   alpha-equivalent. *)
Theorem alpha_let_iff : alpha_let_iff_statement.
Proof.
  intros x x' kind kind' d d' body body' Hclosed Hclosed' Hdistinct
    Hdistinct'.
  split.
  - intros Halpha. apply alpha_equivalent_accepted in Halpha.
    cbn [expr_eqb] in Halpha. apply andb_true_iff in Halpha as [Hd Hbody].
    apply andb_true_iff in Hd as [Hkind Hd]. apply kind_eqb_eq in Hkind.
    apply named_eqb_nil in Hd. split; [ exact Hkind | ].
    split; [ exact Hd | ].
    exists
      ((x :: free_variables_of_expr body)
       ++ x' :: free_variables_of_expr body').
    intros fresh Hfresh. apply among_app_false in Hfresh as [Hfresh Hfresh'].
    assert (Hopens :
      ∀ (bound : variable) (free : list variable),
        among fresh (bound :: free) = false -> opens_at [bound] [fresh] free).
    { intros bound free Houtside.
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ]. intros name Hname.
      destruct (Pos.eqb_spec name fresh) as [-> | Hdifferent];
        [ exact Houtside | ].
      apply Pos.eqb_neq in Hdifferent.
      rewrite among_cons, Hdifferent in Hname. discriminate Hname. }
    apply expr_eqb_alpha_equivalent.
    + rewrite code_closed_equivariant. exact Hclosed.
    + rewrite code_closed_equivariant. exact Hclosed'.
    + rewrite binders_distinct_equivariant. exact Hdistinct.
    + rewrite binders_distinct_equivariant. exact Hdistinct'.
    + apply (opened_eqb [x] [x'] [fresh] [] [] [] body body' [(x, x')]);
        solve [ assumption | apply opens_at_nil | reflexivity
              | apply Hopens; assumption ].
  - intros (<- & <- & Hfresh).
    exact (alpha_let_fresh x x' kind d body body' Hclosed Hclosed' Hfresh).
Qed.
