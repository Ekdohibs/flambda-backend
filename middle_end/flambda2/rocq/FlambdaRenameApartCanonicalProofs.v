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

(* Proofs of the statements of FlambdaRenameApartCanonicalSpec.v.
   Nothing here needs review.

   Alpha-equivalent terms have the same canonical form. What is proved,
   by induction on the derivation of [alpha_equivalent], is of [apart]
   under any renaming and with any new names that are distinct, as
   many as the binders of the term, and outside what the renaming
   makes of the free names of the term ([supplied]): the two terms
   give the same ([same_apart]).

   The rules that rename are instances of one recursion on the term
   ([apart_fused_all], by FlambdaTermInductionProofs.term_mutual_ind):
   the candidate of a renamed term under [r] is the candidate of the
   term under [r'], with the same new names, where [r'] does on the
   free names of the term what [r] does after the renaming of the term
   ([agreeing]) and the new names are outside what [r'] makes of those
   ([outside]). A binder keeps both: the new renaming swaps what the
   bound name is under the renaming so far with the first new name,
   and that moves no other free name of the scope, since the new names
   are outside them ([agreeing_bind_variable],
   [outside_bind_variable]).

   The rule of contexts is by induction on the context ([plug_apart]);
   a term in a list of bindings or of handlers is replaced where it
   stands ([thread_replaced]).

   The supply of the canonical form is the same for alpha-equivalent
   terms ([canonical_supply_alpha]): they have as many binders
   ([alpha_equivalent_binders]) and the same free names.

   The names that a candidate binds are the new names that it takes,
   in order ([apart_binds_all]).

   The canonical form of a term with closed code and distinct binders
   is alpha-equivalent to it ([canonical_avoiding_alpha]). The proof
   of FlambdaRenameApartAlphaProofs.v asks new names outside every
   name of the term, and those of the canonical form are outside its
   free names only. So the term is first renamed apart with new names
   above those of the canonical form too. The result has the canonical
   form of the term, since it is alpha-equivalent to it; its free
   names are those of the term and the names that it binds are the
   new names that it was given, and a name of a term is free in it or
   bound in it ([names_split_all]); so the new names of the canonical
   form are outside every name of the result, and that proof
   applies to the result.

   The free names of what [apart] gives are among what the renaming
   makes of the free names of the term ([apart_free_all]), under
   [supplied] and with no other premise.

   The canonical form of a renamed term
   ([canonical_avoiding_equivariant_up_to]) is as in
   FlambdaRenameApartProofs.rename_apart_avoiding_equivariant_up_to_names,
   with the free names of the term in place of all its names: the
   second renaming is the first and then swaps that fix what it makes
   of the free names and send what it makes of each new name of the
   term to the new name at the same place of the renamed term. It is
   not the first renaming on the names that the term binds, so the
   term renamed by one is not the term renamed by the other; both
   canonical forms are what [apart] gives of the term under the second
   renaming, by [apart_fused_all].

   The candidate of a term with closed code has closed code
   ([apart_closed_all]): the code that it has is what [apart] gives of
   code of the term, whose free names are among what the renaming
   makes of the free names of that code, and those are none. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaCompareSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaContextSpec FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs
  FlambdaFreshQuantifierSpec FlambdaFreshQuantifierProofs
  FlambdaCompareAlphaSpec FlambdaCompareAlphaProofs
  FlambdaRenameApartSpec FlambdaRenameApartProofs
  FlambdaRenameApartAlphaSpec FlambdaRenameApartAlphaProofs
  FlambdaTermInductionProofs FlambdaRenameApartCanonicalSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists of names                                                           *)
(* ======================================================================== *)

Lemma in_without (name : positive) (bound names : list positive) :
  In name names -> In name bound ∨ In name (without bound names).
Proof.
  intros Hin. destruct (among name bound) eqn:Hbound.
  - left. apply among_In, Hbound.
  - right. apply among_In, among_without_all; [ | apply among_In, Hin ].
    intros Hin'. apply among_In in Hin'. rewrite Hin' in Hbound.
    discriminate Hbound.
Qed.

Lemma in_without_one (name bound : positive) (names : list positive) :
  In name names -> name = bound ∨ In name (without [bound] names).
Proof.
  intros Hin.
  destruct (in_without name [bound] names Hin) as [[Heq | []] | Hin'];
    [ left; symmetry; exact Heq | right; exact Hin' ].
Qed.

Lemma without_in (name : positive) (bound names : list positive) :
  In name (without bound names) -> In name names.
Proof. intros Hin. apply among_In in Hin. apply among_without_in in Hin.
  exact (proj1 Hin). Qed.

(* ======================================================================== *)
(* The new names are outside what the renaming makes of some names          *)
(* ======================================================================== *)

Definition outside (r : renaming) (news : list positive)
    (vs : list variable) (ks : list continuation) : Prop :=
  (∀ x : variable, In x vs -> ~ In (rename_variable r x) news)
  ∧ (∀ k : continuation, In k ks -> ~ In (rename_continuation r k) news).

Lemma outside_incl (r : renaming) (news news' : list positive)
    (vs vs' : list variable) (ks ks' : list continuation) :
  outside r news vs ks ->
  incl news' news -> incl vs' vs -> incl ks' ks ->
  outside r news' vs' ks'.
Proof.
  intros [Hv Hk] Hnews Hvs Hks. split.
  - intros x Hx Hin. exact (Hv x (Hvs x Hx) (Hnews _ Hin)).
  - intros k Hk' Hin. exact (Hk k (Hks k Hk') (Hnews _ Hin)).
Qed.

Lemma incl_skipn {A : Type} (n : nat) (l : list A) : incl (skipn n l) l.
Proof. intros a Ha. exact (in_skipn a n l Ha). Qed.

(* A binder takes the first of the new names: the others are outside
   what the new renaming makes of the names of the scope. *)
Lemma outside_bind_variable (r : renaming) (x : variable) (n : positive)
    (news : list positive) (vs vs' : list variable)
    (ks : list continuation) :
  NoDup (n :: news) ->
  outside r (n :: news) vs ks ->
  (∀ y : variable, In y vs' -> y = x ∨ In y vs) ->
  outside (bind_variable r x n) news vs' ks.
Proof.
  intros Hnodup [Hv Hk] Hsub. split.
  - intros y Hy. rewrite rename_variable_bind_variable.
    destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite swap_first. inversion Hnodup; assumption.
    + destruct (Hsub y Hy) as [Heq | Hin]; [ contradiction | ].
      specialize (Hv y Hin). rewrite swap_other.
      * intros Hin'. apply Hv. right. exact Hin'.
      * intros Heq. apply Hne.
        exact (rename_variable_injective r _ _ Heq).
      * intros Heq. apply Hv. left. symmetry. exact Heq.
  - intros k Hin Hin'. apply (Hk k Hin). right. exact Hin'.
Qed.

Lemma outside_bind_continuation (r : renaming) (k : continuation)
    (n : positive) (news : list positive) (vs : list variable)
    (ks ks' : list continuation) :
  NoDup (n :: news) ->
  outside r (n :: news) vs ks ->
  (∀ j : continuation, In j ks' -> j = k ∨ In j ks) ->
  outside (bind_continuation r k n) news vs ks'.
Proof.
  intros Hnodup [Hv Hk] Hsub. split.
  - intros x Hin Hin'. apply (Hv x Hin). right. exact Hin'.
  - intros j Hj. rewrite rename_continuation_bind_continuation.
    destruct (Pos.eq_dec j k) as [-> | Hne].
    + rewrite swap_first. inversion Hnodup; assumption.
    + destruct (Hsub j Hj) as [Heq | Hin]; [ contradiction | ].
      specialize (Hk j Hin). rewrite swap_other.
      * intros Hin'. apply Hk. right. exact Hin'.
      * intros Heq. apply Hne.
        exact (rename_continuation_injective r _ _ Heq).
      * intros Heq. apply Hk. left. symmetry. exact Heq.
Qed.

Lemma outside_bind_variables (xs : list variable) :
  ∀ (r : renaming) (news : list positive) (vs vs' : list variable)
    (ks : list continuation),
    NoDup news ->
    length xs <= length news ->
    outside r news vs ks ->
    (∀ y : variable, In y vs' -> In y xs ∨ In y vs) ->
    outside (snd (bind_variables r news xs)) (skipn (length xs) news) vs' ks.
Proof.
  induction xs as [ | x xs IH];
    intros r news vs vs' ks Hnodup Hlength Hout Hsub.
  - cbn [bind_variables snd length skipn].
    apply (outside_incl r news news vs vs' ks ks Hout);
      [ apply incl_refl | | apply incl_refl ].
    intros y Hy. destruct (Hsub y Hy) as [[] | Hin]. exact Hin.
  - destruct news as [ | n news]; [ cbn [length] in Hlength; lia | ].
    cbn [bind_variables new_variable hd tl length skipn].
    pose proof (IH (bind_variable r x n) news (x :: vs) vs' ks) as IH'.
    destruct (bind_variables (bind_variable r x n) news xs)
      as [[rest xs'] r'].
    cbn [snd] in IH' |- *. apply IH'.
    + inversion Hnodup; assumption.
    + cbn [length] in Hlength. lia.
    + apply (outside_bind_variable r x n news vs (x :: vs) ks Hnodup Hout).
      intros y [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ].
    + intros y Hy.
      destruct (Hsub y Hy) as [[Heq | Hin] | Hin];
        [ right; left; exact Heq | left; exact Hin | right; right; exact Hin ].
Qed.

Lemma outside_bind_continuations (js : list continuation) :
  ∀ (r : renaming) (news : list positive) (vs : list variable)
    (ks ks' : list continuation),
    NoDup news ->
    length js <= length news ->
    outside r news vs ks ->
    (∀ j : continuation, In j ks' -> In j js ∨ In j ks) ->
    outside (snd (bind_continuations r news js)) (skipn (length js) news)
      vs ks'.
Proof.
  induction js as [ | k js IH];
    intros r news vs ks ks' Hnodup Hlength Hout Hsub.
  - cbn [bind_continuations snd length skipn].
    apply (outside_incl r news news vs vs ks ks' Hout);
      [ apply incl_refl | apply incl_refl | ].
    intros j Hj. destruct (Hsub j Hj) as [[] | Hin]. exact Hin.
  - destruct news as [ | n news]; [ cbn [length] in Hlength; lia | ].
    cbn [bind_continuations new_continuation hd tl length skipn].
    pose proof (IH (bind_continuation r k n) news vs (k :: ks) ks') as IH'.
    destruct (bind_continuations (bind_continuation r k n) news js)
      as [[rest js'] r'].
    cbn [snd] in IH' |- *. apply IH'.
    + inversion Hnodup; assumption.
    + cbn [length] in Hlength. lia.
    + apply
        (outside_bind_continuation r k n news vs ks (k :: ks) Hnodup Hout).
      intros j [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ].
    + intros j Hj.
      destruct (Hsub j Hj) as [[Heq | Hin] | Hin];
        [ right; left; exact Heq | left; exact Hin | right; right; exact Hin ].
Qed.

(* ======================================================================== *)
(* Two renamings, one after a renaming of the term                          *)
(* ======================================================================== *)

(* On the names given, [r'] does what [r] does after [p]. *)
Definition agreeing (p r r' : renaming) (vs : list variable)
    (ks : list continuation) : Prop :=
  (∀ x : variable,
    In x vs ->
    rename_variable r' x = rename_variable r (rename_variable p x))
  ∧ (∀ k : continuation,
      In k ks ->
      rename_continuation r' k
      = rename_continuation r (rename_continuation p k)).

Lemma agreeing_incl (p r r' : renaming) (vs vs' : list variable)
    (ks ks' : list continuation) :
  agreeing p r r' vs ks -> incl vs' vs -> incl ks' ks ->
  agreeing p r r' vs' ks'.
Proof.
  intros [Hv Hk] Hvs Hks. split.
  - intros x Hx. exact (Hv x (Hvs x Hx)).
  - intros k Hk'. exact (Hk k (Hks k Hk')).
Qed.

Lemma agreeing_bind_variable (p r r' : renaming) (x : variable)
    (n : positive) (news : list positive) (vs vs' : list variable)
    (ks : list continuation) :
  agreeing p r r' vs ks ->
  outside r' (n :: news) vs ks ->
  (∀ y : variable, In y vs' -> y = x ∨ In y vs) ->
  agreeing p (bind_variable r (rename_variable p x) n) (bind_variable r' x n)
    vs' ks.
Proof.
  intros [Hv Hk] [Hov Hok] Hsub. split; [ | exact Hk ].
  intros y Hy. rewrite !rename_variable_bind_variable.
  destruct (Pos.eq_dec y x) as [-> | Hne].
  - rewrite !swap_first. reflexivity.
  - destruct (Hsub y Hy) as [Heq | Hin]; [ contradiction | ].
    specialize (Hv y Hin). specialize (Hov y Hin).
    rewrite (swap_other _ _ (rename_variable r' y)).
    + rewrite swap_other; [ exact Hv | | ].
      * intros Heq. apply Hne.
        apply (rename_variable_injective p), (rename_variable_injective r).
        exact Heq.
      * rewrite <- Hv. intros Heq. apply Hov. left. symmetry. exact Heq.
    + intros Heq. apply Hne. exact (rename_variable_injective r' _ _ Heq).
    + intros Heq. apply Hov. left. symmetry. exact Heq.
Qed.

Lemma agreeing_bind_continuation (p r r' : renaming) (k : continuation)
    (n : positive) (news : list positive) (vs : list variable)
    (ks ks' : list continuation) :
  agreeing p r r' vs ks ->
  outside r' (n :: news) vs ks ->
  (∀ j : continuation, In j ks' -> j = k ∨ In j ks) ->
  agreeing p (bind_continuation r (rename_continuation p k) n)
    (bind_continuation r' k n) vs ks'.
Proof.
  intros [Hv Hk] [Hov Hok] Hsub. split; [ exact Hv | ].
  intros j Hj. rewrite !rename_continuation_bind_continuation.
  destruct (Pos.eq_dec j k) as [-> | Hne].
  - rewrite !swap_first. reflexivity.
  - destruct (Hsub j Hj) as [Heq | Hin]; [ contradiction | ].
    specialize (Hk j Hin). specialize (Hok j Hin).
    rewrite (swap_other _ _ (rename_continuation r' j)).
    + rewrite swap_other; [ exact Hk | | ].
      * intros Heq. apply Hne.
        apply (rename_continuation_injective p),
          (rename_continuation_injective r).
        exact Heq.
      * rewrite <- Hk. intros Heq. apply Hok. left. symmetry. exact Heq.
    + intros Heq. apply Hne.
      exact (rename_continuation_injective r' _ _ Heq).
    + intros Heq. apply Hok. left. symmetry. exact Heq.
Qed.

Lemma agreeing_bind_variables (xs : list variable) :
  ∀ (p r r' : renaming) (news : list positive) (vs vs' : list variable)
    (ks : list continuation),
    NoDup news ->
    length xs <= length news ->
    agreeing p r r' vs ks ->
    outside r' news vs ks ->
    (∀ y : variable, In y vs' -> In y xs ∨ In y vs) ->
    agreeing p (snd (bind_variables r news (map (rename_variable p) xs)))
      (snd (bind_variables r' news xs)) vs' ks.
Proof.
  induction xs as [ | x xs IH];
    intros p r r' news vs vs' ks Hnodup Hlength Hagree Hout Hsub.
  - cbn [map bind_variables snd].
    apply (agreeing_incl p r r' vs vs' ks ks Hagree); [ | apply incl_refl ].
    intros y Hy. destruct (Hsub y Hy) as [[] | Hin]. exact Hin.
  - destruct news as [ | n news]; [ cbn [length] in Hlength; lia | ].
    cbn [map bind_variables new_variable hd tl].
    pose proof
      (IH p (bind_variable r (rename_variable p x) n) (bind_variable r' x n)
         news (x :: vs) vs' ks)
      as IH'.
    destruct
      (bind_variables (bind_variable r (rename_variable p x) n) news
         (map (rename_variable p) xs))
      as [[rest xs'] r1].
    destruct (bind_variables (bind_variable r' x n) news xs)
      as [[rest' xs''] r1'].
    cbn [snd] in IH' |- *. apply IH'.
    + inversion Hnodup; assumption.
    + cbn [length] in Hlength. lia.
    + apply
        (agreeing_bind_variable p r r' x n news vs (x :: vs) ks Hagree Hout).
      intros y [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ].
    + apply (outside_bind_variable r' x n news vs (x :: vs) ks Hnodup Hout).
      intros y [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ].
    + intros y Hy.
      destruct (Hsub y Hy) as [[Heq | Hin] | Hin];
        [ right; left; exact Heq | left; exact Hin | right; right; exact Hin ].
Qed.

Lemma agreeing_bind_continuations (js : list continuation) :
  ∀ (p r r' : renaming) (news : list positive) (vs : list variable)
    (ks ks' : list continuation),
    NoDup news ->
    length js <= length news ->
    agreeing p r r' vs ks ->
    outside r' news vs ks ->
    (∀ j : continuation, In j ks' -> In j js ∨ In j ks) ->
    agreeing p
      (snd (bind_continuations r news (map (rename_continuation p) js)))
      (snd (bind_continuations r' news js)) vs ks'.
Proof.
  induction js as [ | k js IH];
    intros p r r' news vs ks ks' Hnodup Hlength Hagree Hout Hsub.
  - cbn [map bind_continuations snd].
    apply (agreeing_incl p r r' vs vs ks ks' Hagree); [ apply incl_refl | ].
    intros j Hj. destruct (Hsub j Hj) as [[] | Hin]. exact Hin.
  - destruct news as [ | n news]; [ cbn [length] in Hlength; lia | ].
    cbn [map bind_continuations new_continuation hd tl].
    pose proof
      (IH p (bind_continuation r (rename_continuation p k) n)
         (bind_continuation r' k n) news vs (k :: ks) ks')
      as IH'.
    destruct
      (bind_continuations (bind_continuation r (rename_continuation p k) n)
         news (map (rename_continuation p) js))
      as [[rest js'] r1].
    destruct (bind_continuations (bind_continuation r' k n) news js)
      as [[rest' js''] r1'].
    cbn [snd] in IH' |- *. apply IH'.
    + inversion Hnodup; assumption.
    + cbn [length] in Hlength. lia.
    + apply
        (agreeing_bind_continuation p r r' k n news vs ks (k :: ks) Hagree
           Hout).
      intros j [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ].
    + apply
        (outside_bind_continuation r' k n news vs ks (k :: ks) Hnodup Hout).
      intros j [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ].
    + intros j Hj.
      destruct (Hsub j Hj) as [[Heq | Hin] | Hin];
        [ right; left; exact Heq | left; exact Hin | right; right; exact Hin ].
Qed.

(* What binding takes and leaves, as one equation. *)
Lemma bind_variables_eq (xs : list variable) (r : renaming)
    (news : list positive) :
  length xs <= length news ->
  bind_variables r news xs
  = (skipn (length xs) news, firstn (length xs) news,
     snd (bind_variables r news xs)).
Proof.
  intros Hlength. pose proof (bind_variables_rest xs r news) as Hrest.
  pose proof (bind_variables_taken xs r news Hlength) as Htaken.
  destruct (bind_variables r news xs) as [[rest xs'] r'].
  cbn [fst snd] in Hrest, Htaken |- *. subst. reflexivity.
Qed.

Lemma bind_continuations_eq (ks : list continuation) (r : renaming)
    (news : list positive) :
  length ks <= length news ->
  bind_continuations r news ks
  = (skipn (length ks) news, firstn (length ks) news,
     snd (bind_continuations r news ks)).
Proof.
  intros Hlength. pose proof (bind_continuations_rest ks r news) as Hrest.
  pose proof (bind_continuations_taken ks r news Hlength) as Htaken.
  destruct (bind_continuations r news ks) as [[rest ks'] r'].
  cbn [fst snd] in Hrest, Htaken |- *. subst. reflexivity.
Qed.

(* ======================================================================== *)
(* The candidate of a renamed term, with the same new names                 *)
(* ======================================================================== *)

(* The renamed term under [r] and the term under [r'] have the same
   candidate, where [r'] does what [r] does after the renaming of the
   term on the free names of the term, and the new names are outside
   what [r'] makes of those. With the identity for the renaming of the
   term it says that the candidate depends on the renaming at the free
   names only. *)
Definition apart_fused (e : expr) : Prop :=
  ∀ (p r r' : renaming) (news : list positive),
    NoDup news ->
    binders e <= length news ->
    agreeing p r r' (free_variables_of_expr e)
      (free_continuations_of_expr e) ->
    outside r' news (free_variables_of_expr e)
      (free_continuations_of_expr e) ->
    apart r news (rename_expr p e) = apart r' news e.

Definition apart_handler_fused (h : continuation_handler) : Prop :=
  ∀ (p r r' : renaming) (news : list positive),
    NoDup news ->
    binders_in_handler h <= length news ->
    agreeing p r r' (free_variables_of_handler h)
      (free_continuations_of_handler h) ->
    outside r' news (free_variables_of_handler h)
      (free_continuations_of_handler h) ->
    apart_handler r news (rename_handler p h) = apart_handler r' news h.

Definition apart_binding_fused (binding : static_binding) : Prop :=
  ∀ (p r r' : renaming) (news : list positive),
    NoDup news ->
    binders_in_binding binding <= length news ->
    agreeing p r r' (free_variables_of_binding binding)
      (free_continuations_of_binding binding) ->
    outside r' news (free_variables_of_binding binding)
      (free_continuations_of_binding binding) ->
    apart_binding r news (rename_static_binding p binding)
    = apart_binding r' news binding.

Definition apart_code_fused (c : code) : Prop :=
  ∀ (p r r' : renaming) (news : list positive),
    NoDup news ->
    binders_in_code c <= length news ->
    agreeing p r r' (free_variables_of_code c)
      (free_continuations_of_code c) ->
    outside r' news (free_variables_of_code c)
      (free_continuations_of_code c) ->
    apart_code r news (rename_code p c) = apart_code r' news c.

(* What has no binder is renamed. *)
Lemma renamed_fused (p r r' : renaming) (e : expr) :
  agreeing p r r' (variables_of e) (continuations_of e) ->
  rename_expr r (rename_expr p e) = rename_expr r' e.
Proof.
  intros [Hv Hk]. rewrite <- rename_expr_compose.
  apply rename_expr_agree.
  - intros x Hx. rewrite rename_variable_compose. symmetry. exact (Hv x Hx).
  - intros k Hk'. rewrite rename_continuation_compose. symmetry.
    exact (Hk k Hk').
Qed.

Lemma apply_fused (ap : apply_expr) : apart_fused (Apply ap).
Proof.
  intros p r r' news _ _ Hagree _.
  change (apart r news (rename_expr p (Apply ap)))
    with (news, rename_expr r (rename_expr p (Apply ap))).
  change (apart r' news (Apply ap)) with (news, rename_expr r' (Apply ap)).
  rewrite (renamed_fused p r r' (Apply ap) Hagree). reflexivity.
Qed.

Lemma apply_cont_fused (ac : apply_cont) : apart_fused (Apply_cont ac).
Proof.
  intros p r r' news _ _ Hagree _.
  change (apart r news (rename_expr p (Apply_cont ac)))
    with (news, rename_expr r (rename_expr p (Apply_cont ac))).
  change (apart r' news (Apply_cont ac))
    with (news, rename_expr r' (Apply_cont ac)).
  rewrite (renamed_fused p r r' (Apply_cont ac) Hagree). reflexivity.
Qed.

Lemma switch_fused (scrutinee : simple) (arms : list (Z * apply_cont)) :
  apart_fused (Switch scrutinee arms).
Proof.
  intros p r r' news _ _ Hagree _.
  change (apart r news (rename_expr p (Switch scrutinee arms)))
    with (news, rename_expr r (rename_expr p (Switch scrutinee arms))).
  change (apart r' news (Switch scrutinee arms))
    with (news, rename_expr r' (Switch scrutinee arms)).
  rewrite (renamed_fused p r r' (Switch scrutinee arms) Hagree).
  reflexivity.
Qed.

Lemma invalid_fused : apart_fused Invalid.
Proof. intros p r r' news _ _ _ _. reflexivity. Qed.

Lemma let_fused (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  apart_fused body -> apart_fused (Let x kind d body).
Proof.
  intros IH p r r' news Hnodup Hcount Hagree Hout.
  cbn [binders] in Hcount.
  destruct news as [ | n news]; [ cbn [length] in Hcount; lia | ].
  cbn [free_variables_of_expr free_continuations_of_expr] in Hagree, Hout.
  assert (Hsub :
    ∀ y : variable,
      In y (free_variables_of_expr body) ->
      y = x
      ∨ In y
          (free_variables_of_named d
           ++ without [x] (free_variables_of_expr body))).
  { intros y Hy.
    destruct (in_without_one y x _ Hy) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ]. }
  cbn [rename_expr apart new_variable hd tl].
  rewrite
    (IH p (bind_variable r (rename_variable p x) n) (bind_variable r' x n)
       news).
  - replace (rename_named r (rename_named p d)) with (rename_named r' d);
      [ reflexivity | ].
    rewrite <- rename_named_compose. symmetry. apply rename_named_agree.
    intros y Hy. rewrite rename_variable_compose. symmetry.
    apply (proj1 Hagree), in_or_app. left.
    rewrite free_variables_of_named_variables. exact Hy.
  - inversion Hnodup; assumption.
  - cbn [length] in Hcount. lia.
  - exact (agreeing_bind_variable p r r' x n news _ _ _ Hagree Hout Hsub).
  - exact (outside_bind_variable r' x n news _ _ _ Hnodup Hout Hsub).
Qed.

Lemma handler_fused (params : list variable) (metadata : handler_metadata)
    (handler : expr) :
  apart_fused handler ->
  apart_handler_fused (Handler params metadata handler).
Proof.
  intros IH p r r' news Hnodup Hcount Hagree Hout.
  cbn [binders_in_handler] in Hcount.
  cbn [free_variables_of_handler free_continuations_of_handler]
    in Hagree, Hout.
  assert (Hsub :
    ∀ y : variable,
      In y (free_variables_of_expr handler) ->
      In y params ∨ In y (without params (free_variables_of_expr handler)))
    by (intros y Hy; exact (in_without y params _ Hy)).
  assert (Hlength : length params <= length news) by lia.
  cbn [rename_handler apart_handler].
  rewrite (bind_variables_eq (map (rename_variable p) params) r news)
    by (rewrite length_map; exact Hlength).
  rewrite (bind_variables_eq params r' news Hlength).
  rewrite length_map.
  rewrite
    (IH p (snd (bind_variables r news (map (rename_variable p) params)))
       (snd (bind_variables r' news params)) (skipn (length params) news)).
  - reflexivity.
  - apply NoDup_skipn, Hnodup.
  - rewrite length_skipn. lia.
  - exact
      (agreeing_bind_variables params p r r' news _ _ _ Hnodup Hlength
         Hagree Hout Hsub).
  - exact
      (outside_bind_variables params r' news _ _ _ Hnodup Hlength Hout Hsub).
Qed.

Lemma let_cont_fused (k : continuation) (h : continuation_handler)
    (body : expr) :
  apart_handler_fused h -> apart_fused body ->
  apart_fused (Let_cont_non_recursive k h body).
Proof.
  intros IHh IHbody p r r' news Hnodup Hcount Hagree Hout.
  cbn [binders] in Hcount.
  cbn [free_variables_of_expr free_continuations_of_expr] in Hagree, Hout.
  cbn [rename_expr apart].
  rewrite (IHh p r r' news Hnodup); [ | lia | | ].
  - pose proof (apart_handler_rest h r' news) as Hrest.
    destruct (apart_handler r' news h) as [rest h'].
    cbn [fst] in Hrest. subst rest.
    assert (Hnodup' : NoDup (skipn (binders_in_handler h) news))
      by apply NoDup_skipn, Hnodup.
    assert (Hlength :
      S (binders body) <= length (skipn (binders_in_handler h) news))
      by (rewrite length_skipn; lia).
    assert (Hagree' :
      agreeing p r r' (free_variables_of_expr body)
        (free_continuations_of_handler h
         ++ without [k] (free_continuations_of_expr body))).
    { apply (agreeing_incl _ _ _ _ _ _ _ Hagree);
        [ apply incl_appr | ]; apply incl_refl. }
    assert (Hout' :
      outside r' (skipn (binders_in_handler h) news)
        (free_variables_of_expr body)
        (free_continuations_of_handler h
         ++ without [k] (free_continuations_of_expr body))).
    { apply (outside_incl _ _ _ _ _ _ _ Hout);
        [ apply incl_skipn | apply incl_appr | ]; apply incl_refl. }
    assert (Hsub :
      ∀ j : continuation,
        In j (free_continuations_of_expr body) ->
        j = k
        ∨ In j
            (free_continuations_of_handler h
             ++ without [k] (free_continuations_of_expr body))).
    { intros j Hj.
      destruct (in_without_one j k _ Hj) as [Heq | Hin];
        [ left; exact Heq | right; apply in_or_app; right; exact Hin ]. }
    destruct (skipn (binders_in_handler h) news) as [ | m rest];
      [ cbn [length] in Hlength; lia | ].
    cbn [new_continuation hd tl].
    rewrite
      (IHbody p (bind_continuation r (rename_continuation p k) m)
         (bind_continuation r' k m) rest).
    + reflexivity.
    + inversion Hnodup'; assumption.
    + cbn [length] in Hlength. lia.
    + exact
        (agreeing_bind_continuation p r r' k m rest _ _ _ Hagree' Hout' Hsub).
    + exact
        (outside_bind_continuation r' k m rest _ _ _ Hnodup' Hout' Hsub).
  - apply (agreeing_incl _ _ _ _ _ _ _ Hagree); apply incl_appl, incl_refl.
  - apply (outside_incl _ _ _ _ _ _ _ Hout);
      [ apply incl_refl | | ]; apply incl_appl, incl_refl.
Qed.

(* The handlers of a recursive [Let_cont], gone through in turn. *)
Lemma thread_handlers_fused (p r r' : renaming)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun kh => apart_handler_fused (snd kh)) handlers ->
  ∀ (news : list positive) (vs : list variable) (ks : list continuation),
    NoDup news ->
    list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
    <= length news ->
    agreeing p r r' vs ks ->
    outside r' news vs ks ->
    (∀ kh : continuation * continuation_handler,
      In kh handlers ->
      incl (free_variables_of_handler (snd kh)) vs
      ∧ incl (free_continuations_of_handler (snd kh)) ks) ->
    thread (fun news kh => apart_handler r news (snd kh)) news
      (map (fun '(k, h) => (rename_continuation p k, rename_handler p h))
         handlers)
    = thread (fun news kh => apart_handler r' news (snd kh)) news handlers.
Proof.
  intros Hall.
  induction Hall as [ | [k h] handlers Hh _ IH];
    intros news vs ks Hnodup Hcount Hagree Hout Hincl;
    [ reflexivity | ].
  cbn [map] in Hcount. rewrite list_sum_cons in Hcount. cbn [snd] in Hcount.
  destruct (Hincl (k, h) (or_introl eq_refl)) as [Hvs Hks].
  cbn [snd] in Hh, Hvs, Hks.
  cbn [map thread snd].
  rewrite (Hh p r r' news Hnodup); [ | lia | | ].
  - pose proof (apart_handler_rest h r' news) as Hrest.
    destruct (apart_handler r' news h) as [rest h'].
    cbn [fst] in Hrest. subst rest.
    rewrite (IH (skipn (binders_in_handler h) news) vs ks).
    + reflexivity.
    + apply NoDup_skipn, Hnodup.
    + rewrite length_skipn. lia.
    + exact Hagree.
    + apply (outside_incl _ _ _ _ _ _ _ Hout);
        [ apply incl_skipn | | ]; apply incl_refl.
    + intros kh Hin. apply Hincl. right. exact Hin.
  - exact (agreeing_incl _ _ _ _ _ _ _ Hagree Hvs Hks).
  - exact (outside_incl _ _ _ _ _ _ _ Hout (incl_refl _) Hvs Hks).
Qed.

Lemma let_cont_recursive_fused
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun kh => apart_handler_fused (snd kh)) handlers ->
  apart_fused body ->
  apart_fused (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers IHbody p r r' news Hnodup Hcount Hagree Hout.
  cbn [binders] in Hcount.
  cbn [free_variables_of_expr free_continuations_of_expr] in Hagree, Hout.
  cbn [rename_expr apart]. rewrite map_fst_renamed.
  assert (Hlength : length (map fst handlers) <= length news)
    by (rewrite length_map; lia).
  assert (Hsub :
    ∀ j : continuation,
      In j
        (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers
         ++ free_continuations_of_expr body) ->
      In j (map fst handlers)
      ∨ In j
          (without (map fst handlers)
             (flat_map (fun '(_, h) => free_continuations_of_handler h)
                handlers
              ++ free_continuations_of_expr body)))
    by (intros j Hj; exact (in_without j (map fst handlers) _ Hj)).
  pose proof
    (agreeing_bind_continuations (map fst handlers) p r r' news _ _ _ Hnodup
       Hlength Hagree Hout Hsub)
    as Hagree1.
  pose proof
    (outside_bind_continuations (map fst handlers) r' news _ _ _ Hnodup
       Hlength Hout Hsub)
    as Hout1.
  rewrite
    (bind_continuations_eq (map (rename_continuation p) (map fst handlers))
       r news)
    by (rewrite length_map; exact Hlength).
  rewrite (bind_continuations_eq (map fst handlers) r' news Hlength).
  rewrite (length_map (rename_continuation p)).
  assert (Hnodup1 : NoDup (skipn (length (map fst handlers)) news))
    by apply NoDup_skipn, Hnodup.
  assert (Hcount1 :
    list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
    + binders body
    <= length (skipn (length (map fst handlers)) news))
    by (rewrite length_skipn, length_map; lia).
  set
    (r1 :=
       snd
         (bind_continuations r news
            (map (rename_continuation p) (map fst handlers))))
    in *.
  set (r1' := snd (bind_continuations r' news (map fst handlers))) in *.
  set (rest0 := skipn (length (map fst handlers)) news) in *.
  rewrite
    (thread_handlers_fused p r1 r1' handlers IHhandlers rest0 _ _ Hnodup1
       ltac:(lia) Hagree1 Hout1).
  - pose proof (thread_handlers_rest r1' handlers rest0) as Hrest.
    destruct
      (thread (fun news0 kh => apart_handler r1' news0 (snd kh)) rest0
         handlers)
      as [rest hs'].
    cbn [fst] in Hrest. subst rest.
    rewrite (IHbody p r1 r1' _).
    + reflexivity.
    + apply NoDup_skipn, Hnodup1.
    + rewrite length_skipn. lia.
    + apply (agreeing_incl _ _ _ _ _ _ _ Hagree1);
        apply incl_appr, incl_refl.
    + apply (outside_incl _ _ _ _ _ _ _ Hout1);
        [ apply incl_skipn | | ]; apply incl_appr, incl_refl.
  - intros [k h] Hin. cbn [snd]. split; intros a Ha; apply in_or_app; left;
      apply in_flat_map; exists (k, h); (split; [ exact Hin | exact Ha ]).
Qed.

(* The bindings of a group, gone through in turn. *)
Lemma thread_bindings_fused (p r r' : renaming)
    (bindings : list static_binding) :
  Forall apart_binding_fused bindings ->
  ∀ (news : list positive) (vs : list variable) (ks : list continuation),
    NoDup news ->
    list_sum (map binders_in_binding bindings) <= length news ->
    agreeing p r r' vs ks ->
    outside r' news vs ks ->
    (∀ binding : static_binding,
      In binding bindings ->
      incl (free_variables_of_binding binding) vs
      ∧ incl (free_continuations_of_binding binding) ks) ->
    thread (apart_binding r) news (map (rename_static_binding p) bindings)
    = thread (apart_binding r') news bindings.
Proof.
  intros Hall.
  induction Hall as [ | binding bindings Hb _ IH];
    intros news vs ks Hnodup Hcount Hagree Hout Hincl;
    [ reflexivity | ].
  cbn [map] in Hcount. rewrite list_sum_cons in Hcount.
  destruct (Hincl binding (or_introl eq_refl)) as [Hvs Hks].
  cbn [map thread].
  rewrite (Hb p r r' news Hnodup); [ | lia | | ].
  - pose proof (apart_binding_rest binding r' news) as Hrest.
    destruct (apart_binding r' news binding) as [rest binding'].
    cbn [fst] in Hrest. subst rest.
    rewrite (IH (skipn (binders_in_binding binding) news) vs ks).
    + reflexivity.
    + apply NoDup_skipn, Hnodup.
    + rewrite length_skipn. lia.
    + exact Hagree.
    + apply (outside_incl _ _ _ _ _ _ _ Hout);
        [ apply incl_skipn | | ]; apply incl_refl.
    + intros b Hin. apply Hincl. right. exact Hin.
  - exact (agreeing_incl _ _ _ _ _ _ _ Hagree Hvs Hks).
  - exact (outside_incl _ _ _ _ _ _ _ Hout (incl_refl _) Hvs Hks).
Qed.

Lemma let_static_fused (bindings : list static_binding) (body : expr) :
  Forall apart_binding_fused bindings ->
  apart_fused body ->
  apart_fused (Let_static bindings body).
Proof.
  intros IHbindings IHbody p r r' news Hnodup Hcount Hagree Hout.
  cbn [binders] in Hcount.
  cbn [free_variables_of_expr free_continuations_of_expr] in Hagree, Hout.
  cbn [rename_expr apart].
  rewrite
    (thread_bindings_fused p r r' bindings IHbindings news _ _ Hnodup
       ltac:(lia) Hagree Hout).
  - pose proof (thread_bindings_rest r' bindings news) as Hrest.
    destruct (thread (apart_binding r') news bindings) as [rest bindings'].
    cbn [fst] in Hrest. subst rest.
    rewrite (IHbody p r r' _).
    + reflexivity.
    + apply NoDup_skipn, Hnodup.
    + rewrite length_skipn. lia.
    + apply (agreeing_incl _ _ _ _ _ _ _ Hagree);
        apply incl_appr, incl_refl.
    + apply (outside_incl _ _ _ _ _ _ _ Hout);
        [ apply incl_skipn | | ]; apply incl_appr, incl_refl.
  - intros binding Hin. split; intros a Ha; apply in_or_app; left;
      apply in_flat_map; exists binding; (split; [ exact Hin | exact Ha ]).
Qed.

Lemma code_fused (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr) :
  apart_fused body ->
  apart_code_fused (Code params my_closure ret exn metadata body).
Proof.
  intros IH p r r' news Hnodup Hcount Hagree Hout.
  cbn [binders_in_code] in Hcount.
  cbn [free_variables_of_code free_continuations_of_code] in Hagree, Hout.
  destruct news as [ | n news]; [ cbn [length] in Hcount; lia | ].
  cbn [length] in Hcount.
  cbn [rename_code apart_code new_variable hd tl].
  set (vs0 := without (my_closure :: params) (free_variables_of_expr body))
    in *.
  set (ks0 := without [ret; exn] (free_continuations_of_expr body)) in *.
  assert (Hsub1 :
    ∀ y : variable, In y (my_closure :: vs0) -> y = my_closure ∨ In y vs0)
    by (intros y [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ]).
  pose proof
    (agreeing_bind_variable p r r' my_closure n news vs0 (my_closure :: vs0)
       ks0 Hagree Hout Hsub1)
    as Hagree1.
  pose proof
    (outside_bind_variable r' my_closure n news vs0 (my_closure :: vs0) ks0
       Hnodup Hout Hsub1)
    as Hout1.
  assert (Hnodup1 : NoDup news) by (inversion Hnodup; assumption).
  assert (Hlength : length params <= length news) by lia.
  assert (Hsub2 :
    ∀ y : variable,
      In y (free_variables_of_expr body) ->
      In y params ∨ In y (my_closure :: vs0)).
  { intros y Hy.
    destruct (in_without y (my_closure :: params) _ Hy)
      as [[Heq | Hin] | Hin];
      [ right; left; exact Heq | left; exact Hin | right; right; exact Hin ]. }
  pose proof
    (agreeing_bind_variables params p _ _ news _ _ ks0 Hnodup1 Hlength
       Hagree1 Hout1 Hsub2)
    as Hagree2.
  pose proof
    (outside_bind_variables params _ news _ _ ks0 Hnodup1 Hlength Hout1
       Hsub2)
    as Hout2.
  rewrite
    (bind_variables_eq (map (rename_variable p) params)
       (bind_variable r (rename_variable p my_closure) n) news)
    by (rewrite length_map; exact Hlength).
  rewrite
    (bind_variables_eq params (bind_variable r' my_closure n) news Hlength).
  rewrite length_map.
  assert (Hnodup2 : NoDup (skipn (length params) news))
    by apply NoDup_skipn, Hnodup1.
  assert (Hlength2 :
    S (S (binders body)) <= length (skipn (length params) news))
    by (rewrite length_skipn; lia).
  destruct (skipn (length params) news) as [ | m1 [ | m2 rest]];
    [ cbn [length] in Hlength2; lia | cbn [length] in Hlength2; lia | ].
  cbn [length] in Hlength2.
  cbn [new_continuation hd tl].
  assert (Hsub3 :
    ∀ j : continuation, In j (ret :: ks0) -> j = ret ∨ In j ks0)
    by (intros j [Heq | Hin];
        [ left; symmetry; exact Heq | right; exact Hin ]).
  pose proof
    (agreeing_bind_continuation p _ _ ret m1 (m2 :: rest) _ ks0 (ret :: ks0)
       Hagree2 Hout2 Hsub3)
    as Hagree3.
  pose proof
    (outside_bind_continuation _ ret m1 (m2 :: rest) _ ks0 (ret :: ks0)
       Hnodup2 Hout2 Hsub3)
    as Hout3.
  assert (Hnodup3 : NoDup (m2 :: rest)) by (inversion Hnodup2; assumption).
  assert (Hsub4 :
    ∀ j : continuation,
      In j (free_continuations_of_expr body) ->
      j = exn ∨ In j (ret :: ks0)).
  { intros j Hj.
    destruct (in_without j [ret; exn] _ Hj) as [[Heq | [Heq | []]] | Hin];
      [ right; left; exact Heq | left; symmetry; exact Heq
      | right; right; exact Hin ]. }
  pose proof
    (agreeing_bind_continuation p _ _ exn m2 rest _ _ _ Hagree3 Hout3 Hsub4)
    as Hagree4.
  pose proof
    (outside_bind_continuation _ exn m2 rest _ _ _ Hnodup3 Hout3 Hsub4)
    as Hout4.
  rewrite (IH p _ _ rest ltac:(inversion Hnodup3; assumption) ltac:(lia)
             Hagree4 Hout4).
  reflexivity.
Qed.

Lemma bind_code_fused (id : code_id) (c : code) :
  apart_code_fused c -> apart_binding_fused (Bind_code id c).
Proof.
  intros IH p r r' news Hnodup Hcount Hagree Hout.
  cbn [rename_static_binding apart_binding].
  rewrite (IH p r r' news Hnodup Hcount Hagree Hout). reflexivity.
Qed.

Lemma bind_deleted_code_fused (id : code_id) :
  apart_binding_fused (Bind_deleted_code id).
Proof. intros p r r' news _ _ _ _. reflexivity. Qed.

Lemma bind_symbol_fused (s : symbol) (const : static_const) :
  apart_binding_fused (Bind_symbol s const).
Proof.
  intros p r r' news _ _ [Hv _] _. cbn [rename_static_binding apart_binding].
  rewrite <- rename_static_const_compose.
  rewrite (rename_static_const_agree (compose_renaming r p) r' const);
    [ reflexivity | ].
  intros x Hx. rewrite rename_variable_compose. symmetry. apply Hv.
  destruct const; exact Hx.
Qed.

Lemma apart_fused_all :
  (∀ e : expr, apart_fused e)
  ∧ (∀ h : continuation_handler, apart_handler_fused h)
  ∧ (∀ binding : static_binding, apart_binding_fused binding)
  ∧ (∀ c : code, apart_code_fused c).
Proof.
  apply term_mutual_ind.
  - exact let_fused.
  - exact let_static_fused.
  - exact let_cont_fused.
  - exact let_cont_recursive_fused.
  - exact apply_fused.
  - exact apply_cont_fused.
  - exact switch_fused.
  - exact invalid_fused.
  - exact handler_fused.
  - exact bind_code_fused.
  - exact bind_deleted_code_fused.
  - exact bind_symbol_fused.
  - exact code_fused.
Qed.

(* ======================================================================== *)
(* Alpha-equivalent terms have as many binders                              *)
(* ======================================================================== *)

Lemma plug_binders (c : context) (e e' : expr) :
  binders e = binders e' -> binders (plug c e) = binders (plug c e').
Proof.
  intros Hsame.
  induction c
    as [ | y kind d c IH | bindings c IH
       | before id params my_closure ret exn metadata c IH after body
       | k h c IH
       | k params metadata c IH body | handlers c IH
       | before k params metadata c IH after body ];
    simpl.
  - exact Hsame.
  - rewrite IH. reflexivity.
  - rewrite IH. reflexivity.
  - rewrite !map_app, !list_sum_app. simpl. rewrite IH. reflexivity.
  - rewrite IH. reflexivity.
  - rewrite IH. reflexivity.
  - rewrite IH. reflexivity.
  - rewrite !length_app, !map_app, !list_sum_app. simpl. rewrite IH.
    reflexivity.
Qed.

Lemma alpha_equivalent_binders (e e' : expr) :
  alpha_equivalent e e' -> binders e = binders e'.
Proof.
  intros Halpha.
  induction Halpha
    as [ e | e e' _ IH | e e' e'' _ IH _ IH' | c e e' _ IH
       | s e _ _ | s x kind d body _ _ | s k h body _ _
       | s k params metadata handler body _ _
       | s before k params metadata handler after body _ _
       | s before id c after body _ ].
  - reflexivity.
  - symmetry. exact IH.
  - rewrite IH. exact IH'.
  - apply plug_binders, IH.
  - symmetry. apply binders_renamed.
  - simpl. rewrite binders_renamed. reflexivity.
  - simpl. rewrite binders_renamed. reflexivity.
  - simpl. rewrite length_map, binders_renamed. reflexivity.
  - simpl. rewrite !length_app, !map_app, !list_sum_app. simpl.
    rewrite length_map, binders_renamed. reflexivity.
  - simpl. rewrite !map_app, !list_sum_app. simpl.
    rewrite binders_in_code_renamed. reflexivity.
Qed.

(* ======================================================================== *)
(* New names enough, and outside the free names                             *)
(* ======================================================================== *)

Definition supplied_names (r : renaming) (news : list positive)
    (count : nat) (vs : list variable) (ks : list continuation) : Prop :=
  NoDup news ∧ count <= length news ∧ outside r news vs ks.

(* What is asked of the new names for an expression under a renaming. *)
Definition supplied (r : renaming) (news : list positive) (e : expr)
    : Prop :=
  supplied_names r news (binders e) (free_variables_of_expr e)
    (free_continuations_of_expr e).

Lemma supplied_skipn (m : nat) (r : renaming) (news : list positive)
    (count count' : nat) (vs vs' : list variable)
    (ks ks' : list continuation) :
  supplied_names r news count vs ks ->
  m + count' <= count -> incl vs' vs -> incl ks' ks ->
  supplied_names r (skipn m news) count' vs' ks'.
Proof.
  intros (Hnodup & Hcount & Hout) Hle Hvs Hks.
  split; [ apply NoDup_skipn, Hnodup | split ].
  - rewrite length_skipn. lia.
  - exact (outside_incl _ _ _ _ _ _ _ Hout (incl_skipn m news) Hvs Hks).
Qed.

Lemma supplied_bind_variable (x : variable) (r : renaming) (n : positive)
    (news : list positive) (count count' : nat) (vs vs' : list variable)
    (ks : list continuation) :
  supplied_names r (n :: news) count vs ks ->
  S count' <= count ->
  (∀ y : variable, In y vs' -> y = x ∨ In y vs) ->
  supplied_names (bind_variable r x n) news count' vs' ks.
Proof.
  intros (Hnodup & Hcount & Hout) Hle Hsub.
  split; [ inversion Hnodup; assumption | split ].
  - cbn [length] in Hcount. lia.
  - exact (outside_bind_variable r x n news vs vs' ks Hnodup Hout Hsub).
Qed.

Lemma supplied_bind_continuation (k : continuation) (r : renaming)
    (n : positive) (news : list positive) (count count' : nat)
    (vs : list variable) (ks ks' : list continuation) :
  supplied_names r (n :: news) count vs ks ->
  S count' <= count ->
  (∀ j : continuation, In j ks' -> j = k ∨ In j ks) ->
  supplied_names (bind_continuation r k n) news count' vs ks'.
Proof.
  intros (Hnodup & Hcount & Hout) Hle Hsub.
  split; [ inversion Hnodup; assumption | split ].
  - cbn [length] in Hcount. lia.
  - exact (outside_bind_continuation r k n news vs ks ks' Hnodup Hout Hsub).
Qed.

Lemma supplied_bind_variables (xs : list variable) (r : renaming)
    (news : list positive) (count count' : nat) (vs vs' : list variable)
    (ks : list continuation) :
  supplied_names r news count vs ks ->
  length xs + count' <= count ->
  (∀ y : variable, In y vs' -> In y xs ∨ In y vs) ->
  supplied_names (snd (bind_variables r news xs)) (skipn (length xs) news)
    count' vs' ks.
Proof.
  intros (Hnodup & Hcount & Hout) Hle Hsub.
  split; [ apply NoDup_skipn, Hnodup | split ].
  - rewrite length_skipn. lia.
  - apply (outside_bind_variables xs r news vs vs' ks Hnodup);
      [ lia | exact Hout | exact Hsub ].
Qed.

Lemma supplied_bind_continuations (js : list continuation) (r : renaming)
    (news : list positive) (count count' : nat) (vs : list variable)
    (ks ks' : list continuation) :
  supplied_names r news count vs ks ->
  length js + count' <= count ->
  (∀ j : continuation, In j ks' -> In j js ∨ In j ks) ->
  supplied_names (snd (bind_continuations r news js))
    (skipn (length js) news) count' vs ks'.
Proof.
  intros (Hnodup & Hcount & Hout) Hle Hsub.
  split; [ apply NoDup_skipn, Hnodup | split ].
  - rewrite length_skipn. lia.
  - apply (outside_bind_continuations js r news vs ks ks' Hnodup);
      [ lia | exact Hout | exact Hsub ].
Qed.

(* ======================================================================== *)
(* Two terms with the same candidate, whatever the renaming so far          *)
(* ======================================================================== *)

Definition same_apart (e e' : expr) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied r news e -> apart r news e = apart r news e'.

(* One element of a list gone through in turn is replaced by another
   that gives the same. *)
Lemma thread_replaced {A B : Type}
    (f : list positive -> A -> list positive * B) (before : list A)
    (a a' : A) (after : list A) :
  ∀ news : list positive,
    f (fst (thread f news before)) a = f (fst (thread f news before)) a' ->
    thread f news (before ++ a :: after)
    = thread f news (before ++ a' :: after).
Proof.
  induction before as [ | b before IH]; intros news Hsame.
  - cbn [app thread fst] in Hsame |- *. rewrite Hsame. reflexivity.
  - cbn [app thread] in Hsame |- *.
    destruct (f news b) as [rest b'].
    rewrite (IH rest); [ reflexivity | ].
    destruct (thread f rest before) as [rest' bs']. exact Hsame.
Qed.

(* The body of a handler, and the body of code. *)
Lemma handler_body_apart (params : list variable)
    (metadata : handler_metadata) (body body' : expr) :
  same_apart body body' ->
  ∀ (r : renaming) (news : list positive),
    supplied_names r news
      (binders_in_handler (Handler params metadata body))
      (free_variables_of_handler (Handler params metadata body))
      (free_continuations_of_handler (Handler params metadata body)) ->
    apart_handler r news (Handler params metadata body)
    = apart_handler r news (Handler params metadata body').
Proof.
  intros Hsame r news Hsupplied.
  cbn [binders_in_handler free_variables_of_handler
       free_continuations_of_handler]
    in Hsupplied.
  assert (Hlength : length params <= length news)
    by (destruct Hsupplied as (_ & Hcount & _); lia).
  cbn [apart_handler].
  rewrite (bind_variables_eq params r news Hlength). cbv beta iota.
  rewrite
    (Hsame (snd (bind_variables r news params)) (skipn (length params) news));
    [ reflexivity | ].
  apply
    (supplied_bind_variables params r news _ _ _ _ _ Hsupplied (le_n _)).
  intros y Hy. exact (in_without y params _ Hy).
Qed.

Lemma code_body_apart (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (body body' : expr) :
  same_apart body body' ->
  ∀ (r : renaming) (news : list positive),
    supplied_names r news
      (binders_in_code (Code params my_closure ret exn metadata body))
      (free_variables_of_code (Code params my_closure ret exn metadata body))
      (free_continuations_of_code
         (Code params my_closure ret exn metadata body)) ->
    apart_code r news (Code params my_closure ret exn metadata body)
    = apart_code r news (Code params my_closure ret exn metadata body').
Proof.
  intros Hsame r news Hsupplied.
  cbn [binders_in_code free_variables_of_code free_continuations_of_code]
    in Hsupplied.
  destruct news as [ | n news];
    [ destruct Hsupplied as (_ & Hcount & _); cbn [length] in Hcount; lia
    | ].
  cbn [apart_code new_variable hd tl].
  pose proof
    (supplied_bind_variable my_closure r n news _
       (length params + S (S (binders body))) _
       (my_closure
        :: without (my_closure :: params) (free_variables_of_expr body))
       _ Hsupplied (le_n _)
       ltac:(intros y [Heq | Hin];
             [ left; symmetry; exact Heq | right; exact Hin ]))
    as H1.
  assert (Hlength : length params <= length news)
    by (destruct H1 as (_ & Hcount & _); lia).
  pose proof
    (supplied_bind_variables params _ news _ (S (S (binders body))) _
       (free_variables_of_expr body) _ H1 (le_n _))
    as H2.
  rewrite
    (bind_variables_eq params (bind_variable r my_closure n) news Hlength).
  cbv beta iota.
  assert (H2' := H2 ltac:(
    intros y Hy;
    destruct (in_without y (my_closure :: params) _ Hy)
      as [[Heq | Hin] | Hin];
    [ right; left; exact Heq | left; exact Hin | right; right; exact Hin ])).
  clear H2.
  destruct (skipn (length params) news) as [ | m1 [ | m2 rest]];
    [ destruct H2' as (_ & Hcount & _); cbn [length] in Hcount; lia
    | destruct H2' as (_ & Hcount & _); cbn [length] in Hcount; lia | ].
  cbn [new_continuation hd tl].
  pose proof
    (supplied_bind_continuation ret _ m1 (m2 :: rest) _ (S (binders body)) _
       _ (ret :: without [ret; exn] (free_continuations_of_expr body)) H2'
       (le_n _)
       ltac:(intros j [Heq | Hin];
             [ left; symmetry; exact Heq | right; exact Hin ]))
    as H3.
  pose proof
    (supplied_bind_continuation exn _ m2 rest _ (binders body) _ _
       (free_continuations_of_expr body) H3 (le_n _)
       ltac:(intros j Hj;
             destruct (in_without j [ret; exn] _ Hj)
               as [[Heq | [Heq | []]] | Hin];
             [ right; left; exact Heq | left; symmetry; exact Heq
             | right; right; exact Hin ]))
    as H4.
  rewrite (Hsame _ rest H4). reflexivity.
Qed.

(* A context. *)
Lemma plug_apart (c : context) (e e' : expr) :
  same_apart e e' -> same_apart (plug c e) (plug c e').
Proof.
  intros Hsame.
  induction c
    as [ | y kind d c IH | bindings c IH
       | before id params my_closure ret exn metadata c IH after body
       | k h c IH
       | k params metadata c IH body | handlers c IH
       | before k params metadata c IH after body ];
    cbn [plug]; [ exact Hsame | .. ]; intros r news Hsupplied;
    unfold supplied in Hsupplied;
    cbn [binders free_variables_of_expr free_continuations_of_expr]
      in Hsupplied.
  - destruct news as [ | n news];
      [ destruct Hsupplied as (_ & Hcount & _); cbn [length] in Hcount; lia
      | ].
    cbn [apart new_variable hd tl].
    rewrite (IH (bind_variable r y n) news); [ reflexivity | ].
    apply (supplied_bind_variable y r n news _ _ _ _ _ Hsupplied (le_n _)).
    intros x Hx.
    destruct (in_without_one x y _ Hx) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ].
  - cbn [apart].
    pose proof (thread_bindings_rest r bindings news) as Hrest.
    destruct (thread (apart_binding r) news bindings) as [rest bindings'].
    cbn [fst] in Hrest. subst rest.
    rewrite (IH r (skipn (list_sum (map binders_in_binding bindings)) news));
      [ reflexivity | ].
    apply (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _));
      apply incl_appr, incl_refl.
  - cbn [apart].
    rewrite
      (thread_replaced (apart_binding r) before
         (Bind_code id
            (Code params my_closure ret exn metadata (plug c e)))
         (Bind_code id
            (Code params my_closure ret exn metadata (plug c e')))
         after news);
      [ reflexivity | ].
    rewrite (thread_bindings_rest r before news). cbn [apart_binding].
    rewrite
      (code_body_apart params my_closure ret exn metadata (plug c e)
         (plug c e') IH r _);
      [ reflexivity | ].
    apply (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied).
    + rewrite map_app, list_sum_app. cbn [map binders_in_binding].
      rewrite list_sum_cons. lia.
    + intros x Hx. apply in_or_app. left. apply in_flat_map.
      exists
        (Bind_code id (Code params my_closure ret exn metadata (plug c e))).
      split; [ apply in_or_app; right; left; reflexivity | exact Hx ].
    + intros j Hj. apply in_or_app. left. apply in_flat_map.
      exists
        (Bind_code id (Code params my_closure ret exn metadata (plug c e))).
      split; [ apply in_or_app; right; left; reflexivity | exact Hj ].
  - cbn [apart].
    pose proof (apart_handler_rest h r news) as Hrest.
    destruct (apart_handler r news h) as [rest h'].
    cbn [fst] in Hrest. subst rest.
    pose proof
      (supplied_skipn (binders_in_handler h) r news _
         (S (binders (plug c e))) _ (free_variables_of_expr (plug c e)) _
         (free_continuations_of_handler h
          ++ without [k] (free_continuations_of_expr (plug c e)))
         Hsupplied (le_n _) ltac:(apply incl_appr, incl_refl)
         ltac:(apply incl_refl))
      as H1.
    destruct (skipn (binders_in_handler h) news) as [ | m rest];
      [ destruct H1 as (_ & Hcount & _); cbn [length] in Hcount; lia | ].
    cbn [new_continuation hd tl].
    rewrite (IH (bind_continuation r k m) rest); [ reflexivity | ].
    apply (supplied_bind_continuation k r m rest _ _ _ _ _ H1 (le_n _)).
    intros j Hj.
    destruct (in_without_one j k _ Hj) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ].
  - cbn [apart].
    rewrite
      (handler_body_apart params metadata (plug c e) (plug c e') IH r news);
      [ reflexivity | ].
    apply (supplied_skipn 0 r news _ _ _ _ _ _ Hsupplied);
      [ lia | apply incl_appl, incl_refl | apply incl_appl, incl_refl ].
  - cbn [apart].
    assert (Hlength : length (map fst handlers) <= length news)
      by (destruct Hsupplied as (_ & Hcount & _); rewrite length_map; lia).
    rewrite (bind_continuations_eq (map fst handlers) r news Hlength).
    cbv beta iota.
    pose proof
      (supplied_bind_continuations (map fst handlers) r news _
         (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
          + binders (plug c e))
         _ _
         (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers
          ++ free_continuations_of_expr (plug c e))
         Hsupplied ltac:(rewrite length_map; lia)
         ltac:(intros j Hj; exact (in_without j (map fst handlers) _ Hj)))
      as H1.
    set (r1 := snd (bind_continuations r news (map fst handlers))) in *.
    set (rest0 := skipn (length (map fst handlers)) news) in *.
    pose proof (thread_handlers_rest r1 handlers rest0) as Hrest.
    destruct
      (thread (fun news0 kh => apart_handler r1 news0 (snd kh)) rest0
         handlers)
      as [rest hs'].
    cbn [fst] in Hrest. subst rest.
    rewrite (IH r1 _); [ reflexivity | ].
    apply (supplied_skipn _ r1 rest0 _ _ _ _ _ _ H1 (le_n _));
      apply incl_appr, incl_refl.
  - assert (Hks :
      map fst
        (before ++ (k, Handler params metadata (plug c e')) :: after)
      = map fst
          (before ++ (k, Handler params metadata (plug c e)) :: after))
      by (rewrite !map_app; reflexivity).
    cbn [apart]. rewrite Hks.
    assert (Hlength :
      length
        (map fst
           (before ++ (k, Handler params metadata (plug c e)) :: after))
      <= length news)
      by (destruct Hsupplied as (_ & Hcount & _); rewrite length_map; lia).
    rewrite (bind_continuations_eq _ r news Hlength). cbv beta iota.
    pose proof
      (supplied_bind_continuations
         (map fst
            (before ++ (k, Handler params metadata (plug c e)) :: after))
         r news _
         (list_sum
            (map (fun kh => binders_in_handler (snd kh))
               (before ++ (k, Handler params metadata (plug c e)) :: after))
          + binders body)
         _ _
         (flat_map (fun '(_, h) => free_continuations_of_handler h)
            (before ++ (k, Handler params metadata (plug c e)) :: after)
          ++ free_continuations_of_expr body)
         Hsupplied ltac:(rewrite length_map; lia)
         ltac:(intros j Hj; exact (in_without j _ _ Hj)))
      as H1.
    set
      (r1 :=
         snd
           (bind_continuations r news
              (map fst
                 (before
                  ++ (k, Handler params metadata (plug c e)) :: after))))
      in *.
    set
      (rest0 :=
         skipn
           (length
              (map fst
                 (before
                  ++ (k, Handler params metadata (plug c e)) :: after)))
           news)
      in *.
    rewrite
      (thread_replaced (fun news0 kh => apart_handler r1 news0 (snd kh))
         before (k, Handler params metadata (plug c e))
         (k, Handler params metadata (plug c e')) after rest0);
      [ reflexivity | ].
    rewrite (thread_handlers_rest r1 before rest0). cbn [snd].
    apply (handler_body_apart params metadata (plug c e) (plug c e') IH r1).
    apply (supplied_skipn _ r1 rest0 _ _ _ _ _ _ H1).
    + rewrite map_app, list_sum_app. cbn [map snd]. rewrite list_sum_cons.
      lia.
    + intros x Hx. apply in_or_app. left. apply in_flat_map.
      exists (k, Handler params metadata (plug c e)).
      split; [ apply in_or_app; right; left; reflexivity | exact Hx ].
    + intros j Hj. apply in_or_app. left. apply in_flat_map.
      exists (k, Handler params metadata (plug c e)).
      split; [ apply in_or_app; right; left; reflexivity | exact Hj ].
Qed.

(* ======================================================================== *)
(* The rules that rename                                                    *)
(* ======================================================================== *)

Lemma fixes_agreeing (s r : renaming) (vs : list variable)
    (ks : list continuation) :
  fixes_names s vs ks -> agreeing s r r vs ks.
Proof.
  intros [Hv Hk]. split.
  - intros x Hx.
    rewrite (fixes_variable s vs x Hv (proj2 (among_In x vs) Hx)).
    reflexivity.
  - intros k Hk'.
    rewrite (fixes_continuation s ks k Hk (proj2 (among_In k ks) Hk')).
    reflexivity.
Qed.

Lemma renamed_apart (s : renaming) (e : expr) :
  fixes_names s (free_variables_of_expr e) (free_continuations_of_expr e) ->
  same_apart e (rename_expr s e).
Proof.
  intros Hfixes r news (Hnodup & Hcount & Hout). symmetry.
  exact
    (proj1 apart_fused_all e s r r news Hnodup Hcount
       (fixes_agreeing s r _ _ Hfixes) Hout).
Qed.

Lemma let_renamed_apart (s : renaming) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  fixes_names s (without [x] (free_variables_of_expr body))
    (free_continuations_of_expr body) ->
  same_apart (Let x kind d body)
    (Let (rename_variable s x) kind d (rename_expr s body)).
Proof.
  intros Hfixes r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  destruct news as [ | n news];
    [ destruct Hsupplied as (_ & Hcount & _); cbn [length] in Hcount; lia
    | ].
  assert (Hsub :
    ∀ y : variable,
      In y (free_variables_of_expr body) ->
      y = x ∨ In y (without [x] (free_variables_of_expr body)))
    by (intros y Hy; exact (in_without_one y x _ Hy)).
  destruct
    (supplied_skipn 0 r (n :: news) _ (S (binders body)) _
       (without [x] (free_variables_of_expr body)) _
       (free_continuations_of_expr body) Hsupplied (le_n _)
       ltac:(apply incl_appr, incl_refl) ltac:(apply incl_refl))
    as (Hnodup & Hcount & Hout).
  cbn [skipn] in Hnodup, Hcount, Hout.
  cbn [apart new_variable hd tl].
  rewrite
    (proj1 apart_fused_all body s (bind_variable r (rename_variable s x) n)
       (bind_variable r x n) news).
  - reflexivity.
  - inversion Hnodup; assumption.
  - cbn [length] in Hcount. lia.
  - exact
      (agreeing_bind_variable s r r x n news _ _ _
         (fixes_agreeing s r _ _ Hfixes) Hout Hsub).
  - exact (outside_bind_variable r x n news _ _ _ Hnodup Hout Hsub).
Qed.

Lemma let_cont_renamed_apart (s : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  fixes_names s (free_variables_of_expr body)
    (without [k] (free_continuations_of_expr body)) ->
  same_apart (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive (rename_continuation s k) h
       (rename_expr s body)).
Proof.
  intros Hfixes r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  cbn [apart].
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst] in Hrest. subst rest.
  assert (Hsub :
    ∀ j : continuation,
      In j (free_continuations_of_expr body) ->
      j = k ∨ In j (without [k] (free_continuations_of_expr body)))
    by (intros j Hj; exact (in_without_one j k _ Hj)).
  destruct
    (supplied_skipn (binders_in_handler h) r news _ (S (binders body)) _
       (free_variables_of_expr body) _
       (without [k] (free_continuations_of_expr body)) Hsupplied (le_n _)
       ltac:(apply incl_appr, incl_refl) ltac:(apply incl_appr, incl_refl))
    as (Hnodup & Hcount & Hout).
  destruct (skipn (binders_in_handler h) news) as [ | m rest];
    [ cbn [length] in Hcount; lia | ].
  cbn [new_continuation hd tl].
  rewrite
    (proj1 apart_fused_all body s
       (bind_continuation r (rename_continuation s k) m)
       (bind_continuation r k m) rest).
  - reflexivity.
  - inversion Hnodup; assumption.
  - cbn [length] in Hcount. lia.
  - exact
      (agreeing_bind_continuation s r r k m rest _ _ _
         (fixes_agreeing s r _ _ Hfixes) Hout Hsub).
  - exact (outside_bind_continuation r k m rest _ _ _ Hnodup Hout Hsub).
Qed.

Lemma handler_renamed_apart (s : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  fixes_names s
    (free_variables_of_handler (Handler params metadata handler))
    (free_continuations_of_expr handler) ->
  same_apart
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k
       (rename_handler s (Handler params metadata handler)) body).
Proof.
  intros Hfixes r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  destruct
    (supplied_skipn 0 r news _
       (binders_in_handler (Handler params metadata handler)) _
       (free_variables_of_handler (Handler params metadata handler)) _
       (free_continuations_of_handler (Handler params metadata handler))
       Hsupplied ltac:(lia) ltac:(apply incl_appl, incl_refl)
       ltac:(apply incl_appl, incl_refl))
    as (Hnodup & Hcount & Hout).
  cbn [skipn] in Hnodup, Hcount, Hout.
  cbn [apart].
  rewrite
    (proj1 (proj2 apart_fused_all) (Handler params metadata handler) s r r
       news Hnodup Hcount (fixes_agreeing s r _ _ Hfixes) Hout).
  reflexivity.
Qed.

Lemma recursive_handler_renamed_apart (s : renaming)
    (before : list (continuation * continuation_handler))
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr)
    (after : list (continuation * continuation_handler)) (body : expr) :
  fixes_names s
    (free_variables_of_handler (Handler params metadata handler))
    (free_continuations_of_expr handler) ->
  same_apart
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
    (Let_cont_recursive
       (before
        ++ (k, rename_handler s (Handler params metadata handler)) :: after)
       body).
Proof.
  intros Hfixes r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  assert (Hks :
    map fst
      (before
       ++ (k, rename_handler s (Handler params metadata handler)) :: after)
    = map fst (before ++ (k, Handler params metadata handler) :: after))
    by (rewrite !map_app; reflexivity).
  cbn [apart]. rewrite Hks.
  assert (Hlength :
    length
      (map fst (before ++ (k, Handler params metadata handler) :: after))
    <= length news)
    by (destruct Hsupplied as (_ & Hcount & _); rewrite length_map; lia).
  rewrite (bind_continuations_eq _ r news Hlength). cbv beta iota.
  pose proof
    (supplied_bind_continuations
       (map fst (before ++ (k, Handler params metadata handler) :: after))
       r news _
       (list_sum
          (map (fun kh => binders_in_handler (snd kh))
             (before ++ (k, Handler params metadata handler) :: after))
        + binders body)
       _ _
       (flat_map (fun '(_, h) => free_continuations_of_handler h)
          (before ++ (k, Handler params metadata handler) :: after)
        ++ free_continuations_of_expr body)
       Hsupplied ltac:(rewrite length_map; lia)
       ltac:(intros j Hj; exact (in_without j _ _ Hj)))
    as H1.
  set
    (r1 :=
       snd
         (bind_continuations r news
            (map fst
               (before ++ (k, Handler params metadata handler) :: after))))
    in *.
  set
    (rest0 :=
       skipn
         (length
            (map fst
               (before ++ (k, Handler params metadata handler) :: after)))
         news)
    in *.
  destruct
    (supplied_skipn
       (list_sum (map (fun kh => binders_in_handler (snd kh)) before)) r1
       rest0 _ (binders_in_handler (Handler params metadata handler)) _
       (free_variables_of_handler (Handler params metadata handler)) _
       (free_continuations_of_handler (Handler params metadata handler)) H1)
    as (Hnodup & Hcount & Hout).
  { rewrite map_app, list_sum_app. cbn [map snd]. rewrite list_sum_cons.
    lia. }
  { intros x Hx. apply in_or_app. left. apply in_flat_map.
    exists (k, Handler params metadata handler).
    split; [ apply in_or_app; right; left; reflexivity | exact Hx ]. }
  { intros j Hj. apply in_or_app. left. apply in_flat_map.
    exists (k, Handler params metadata handler).
    split; [ apply in_or_app; right; left; reflexivity | exact Hj ]. }
  rewrite
    (thread_replaced (fun news0 kh => apart_handler r1 news0 (snd kh)) before
       (k, Handler params metadata handler)
       (k, rename_handler s (Handler params metadata handler)) after rest0);
    [ reflexivity | ].
  rewrite (thread_handlers_rest r1 before rest0). cbn [snd]. symmetry.
  exact
    (proj1 (proj2 apart_fused_all) (Handler params metadata handler) s r1 r1
       _ Hnodup Hcount (fixes_agreeing s r1 _ _ Hfixes) Hout).
Qed.

Lemma code_closed_no_free_name (c : code) :
  code_is_closed c = true ->
  free_variables_of_code c = [] ∧ free_continuations_of_code c = [].
Proof.
  destruct c as [params my_closure ret exn metadata body]. intros Hclosed.
  cbn [code_is_closed] in Hclosed.
  apply andb_true_iff in Hclosed as [Hclosed _].
  apply andb_true_iff in Hclosed as [Hv Hk].
  split.
  - destruct
      (free_variables_of_code
         (Code params my_closure ret exn metadata body));
      [ reflexivity | discriminate Hv ].
  - destruct
      (free_continuations_of_code
         (Code params my_closure ret exn metadata body));
      [ reflexivity | discriminate Hk ].
Qed.

Lemma code_renamed_apart (s : renaming) (before : list static_binding)
    (id : code_id) (c : code) (after : list static_binding) (body : expr) :
  code_is_closed c = true ->
  same_apart (Let_static (before ++ Bind_code id c :: after) body)
    (Let_static (before ++ Bind_code id (rename_code s c) :: after) body).
Proof.
  intros Hclosed r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  destruct
    (supplied_skipn (list_sum (map binders_in_binding before)) r news _
       (binders_in_code c) _ (free_variables_of_code c) _
       (free_continuations_of_code c) Hsupplied)
    as (Hnodup & Hcount & Hout).
  { rewrite map_app, list_sum_app. cbn [map binders_in_binding].
    rewrite list_sum_cons. lia. }
  { intros x Hx. apply in_or_app. left. apply in_flat_map.
    exists (Bind_code id c).
    split; [ apply in_or_app; right; left; reflexivity | exact Hx ]. }
  { intros j Hj. apply in_or_app. left. apply in_flat_map.
    exists (Bind_code id c).
    split; [ apply in_or_app; right; left; reflexivity | exact Hj ]. }
  cbn [apart].
  rewrite
    (thread_replaced (apart_binding r) before (Bind_code id c)
       (Bind_code id (rename_code s c)) after news);
    [ reflexivity | ].
  rewrite (thread_bindings_rest r before news). cbn [apart_binding].
  rewrite (proj2 (proj2 (proj2 apart_fused_all)) c s r r _ Hnodup Hcount);
    [ reflexivity | | exact Hout ].
  destruct (code_closed_no_free_name c Hclosed) as [Hv Hk].
  rewrite Hv, Hk. split; intros a [].
Qed.

(* ======================================================================== *)
(* Alpha-equivalent terms have the same candidate                           *)
(* ======================================================================== *)

Lemma supplied_alpha (e e' : expr) (r : renaming) (news : list positive) :
  alpha_equivalent e e' -> supplied r news e -> supplied r news e'.
Proof.
  intros Halpha (Hnodup & Hcount & Hout). split; [ exact Hnodup | split ].
  - rewrite <- (alpha_equivalent_binders e e' Halpha). exact Hcount.
  - apply (outside_incl _ _ _ _ _ _ _ Hout (incl_refl _)).
    + intros x Hx. apply among_In.
      rewrite (alpha_equivalent_free_variables e e' Halpha x).
      apply among_In, Hx.
    + intros k Hk. apply among_In.
      rewrite (alpha_equivalent_free_continuations e e' Halpha k).
      apply among_In, Hk.
Qed.

Lemma alpha_same_apart (e e' : expr) :
  alpha_equivalent e e' -> same_apart e e'.
Proof.
  intros Halpha.
  induction Halpha
    as [ e | e e' Halpha IH | e e' e'' Halpha IH Halpha' IH'
       | c e e' Halpha IH
       | s e Hclosed Hfixes | s x kind d body Hclosed Hfixes
       | s k h body Hclosed Hfixes
       | s k params metadata handler body Hclosed Hfixes
       | s before k params metadata handler after body Hclosed Hfixes
       | s before id c after body Hclosed ].
  - intros r news _. reflexivity.
  - intros r news Hsupplied. symmetry. apply IH.
    exact (supplied_alpha e' e r news (Alpha_sym e e' Halpha) Hsupplied).
  - intros r news Hsupplied. rewrite (IH r news Hsupplied). apply IH'.
    exact (supplied_alpha e e' r news Halpha Hsupplied).
  - exact (plug_apart c e e' IH).
  - exact (renamed_apart s e Hfixes).
  - exact (let_renamed_apart s x kind d body Hfixes).
  - exact (let_cont_renamed_apart s k h body Hfixes).
  - exact (handler_renamed_apart s k params metadata handler body Hfixes).
  - exact
      (recursive_handler_renamed_apart s before k params metadata handler
         after body Hfixes).
  - exact (code_renamed_apart s before id c after body Hclosed).
Qed.

(* The supply of new names is the same for alpha-equivalent terms. *)
Lemma greatest_in (names : list positive) :
  fold_right Pos.max 1%positive names = 1%positive
  ∨ In (fold_right Pos.max 1%positive names) names.
Proof.
  induction names as [ | a names IH]; [ left; reflexivity | ].
  cbn [fold_right].
  destruct (Pos.max_spec a (fold_right Pos.max 1%positive names))
    as [[_ Heq] | [_ Heq]];
    rewrite Heq.
  - destruct IH as [IH | IH]; [ left; exact IH | right; right; exact IH ].
  - right. left. reflexivity.
Qed.

Lemma greatest_le (names names' : list positive) :
  incl names names' ->
  (fold_right Pos.max 1 names <= fold_right Pos.max 1 names')%positive.
Proof.
  intros Hincl. destruct (greatest_in names) as [Heq | Hin].
  - rewrite Heq. apply Pos.le_1_l.
  - apply fresh_name_above, Hincl, Hin.
Qed.

Lemma same_names_fresh_name (names names' : list positive) :
  same_names names names' -> fresh_name names = fresh_name names'.
Proof.
  intros Hsame. unfold fresh_name. f_equal.
  apply Pos.le_antisym; apply greatest_le; intros a Ha; apply among_In;
    [ rewrite <- Hsame | rewrite Hsame ]; apply among_In, Ha.
Qed.

Lemma canonical_supply_alpha (avoid : list continuation) (e e' : expr) :
  alpha_equivalent e e' ->
  fresh_names (binders e)
    (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e)
  = fresh_names (binders e')
      (avoid ++ free_variables_of_expr e' ++ free_continuations_of_expr e').
Proof.
  intros Halpha. rewrite (alpha_equivalent_binders e e' Halpha).
  unfold fresh_names. f_equal. apply same_names_fresh_name. intros name.
  rewrite !among_app, (alpha_equivalent_free_variables e e' Halpha name),
    (alpha_equivalent_free_continuations e e' Halpha name).
  reflexivity.
Qed.

Lemma canonical_supplied (avoid : list continuation) (e : expr) :
  supplied identity_renaming
    (fresh_names (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    e.
Proof.
  destruct
    (fresh_names_fresh (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    as [Hlength [Hnodup Houtside]].
  split; [ apply distinct_NoDup, Hnodup | ].
  split; [ rewrite Hlength; apply le_n | ].
  split.
  - intros x Hx Hin.
    pose proof (Houtside x (proj2 (among_In _ _) Hin)) as Hnot.
    rewrite !among_app, (proj2 (among_In x _) Hx), orb_true_r in Hnot.
    discriminate Hnot.
  - intros k Hk Hin.
    pose proof (Houtside k (proj2 (among_In _ _) Hin)) as Hnot.
    rewrite !among_app, (proj2 (among_In k _) Hk), !orb_true_r in Hnot.
    discriminate Hnot.
Qed.

Theorem alpha_equivalent_canonical : alpha_equivalent_canonical_statement.
Proof.
  intros avoid e e' Halpha. unfold canonical_avoiding.
  rewrite <- (canonical_supply_alpha avoid e e' Halpha).
  rewrite
    (alpha_same_apart e e' Halpha identity_renaming _
       (canonical_supplied avoid e)).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The names that the candidate binds                                       *)
(* ======================================================================== *)

Lemma firstn_plus {A : Type} (m n : nat) (l : list A) :
  firstn (m + n) l = firstn m l ++ firstn n (skipn m l).
Proof.
  revert l. induction m as [ | m IH]; intros l; [ reflexivity | ].
  destruct l as [ | a l]; cbn [Nat.add firstn skipn app].
  - rewrite firstn_nil. reflexivity.
  - rewrite IH. reflexivity.
Qed.

Lemma thread_length {A B : Type}
    (f : list positive -> A -> list positive * B) (l : list A) :
  ∀ news : list positive, length (snd (thread f news l)) = length l.
Proof.
  induction l as [ | a l IH]; intros news; [ reflexivity | ].
  cbn [thread]. destruct (f news a) as [rest b]. specialize (IH rest).
  destruct (thread f rest l) as [rest' bs]. cbn [snd length] in IH |- *.
  rewrite IH. reflexivity.
Qed.

Lemma combine_fst {A B : Type} (l : list A) :
  ∀ l' : list B, length l = length l' -> map fst (combine l l') = l.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] Hlength;
    try reflexivity; try discriminate Hlength.
  cbn [combine map fst]. rewrite IH; [ reflexivity | ].
  injection Hlength as Hlength. exact Hlength.
Qed.

Lemma combine_snd {A B : Type} (l : list A) :
  ∀ l' : list B, length l = length l' -> map snd (combine l l') = l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] Hlength;
    try reflexivity; try discriminate Hlength.
  cbn [combine map snd]. rewrite IH; [ reflexivity | ].
  injection Hlength as Hlength. exact Hlength.
Qed.

Lemma flat_map_after {A B C : Type} (f : B -> list C) (g : A -> B)
    (l : list A) :
  flat_map (fun a => f (g a)) l = flat_map f (map g l).
Proof.
  induction l as [ | a l IH]; [ reflexivity | ].
  cbn [flat_map map]. rewrite IH. reflexivity.
Qed.

(* The names that the candidate binds are the new names that it takes,
   in the order in which it takes them. *)
Definition apart_binds (e : expr) : Prop :=
  ∀ (r : renaming) (news : list positive),
    binders e <= length news ->
    bound_names (snd (apart r news e)) = firstn (binders e) news.

Definition apart_handler_binds (h : continuation_handler) : Prop :=
  ∀ (r : renaming) (news : list positive),
    binders_in_handler h <= length news ->
    bound_names_in_handler (snd (apart_handler r news h))
    = firstn (binders_in_handler h) news.

Definition apart_binding_binds (binding : static_binding) : Prop :=
  ∀ (r : renaming) (news : list positive),
    binders_in_binding binding <= length news ->
    bound_names_in_binding (snd (apart_binding r news binding))
    = firstn (binders_in_binding binding) news.

Definition apart_code_binds (c : code) : Prop :=
  ∀ (r : renaming) (news : list positive),
    binders_in_code c <= length news ->
    bound_names_in_code (snd (apart_code r news c))
    = firstn (binders_in_code c) news.

Lemma let_binds (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  apart_binds body -> apart_binds (Let x kind d body).
Proof.
  intros IH r news Hcount. cbn [binders] in Hcount |- *.
  destruct news as [ | n news]; [ cbn [length] in Hcount; lia | ].
  cbn [length] in Hcount. cbn [apart new_variable hd tl].
  specialize (IH (bind_variable r x n) news ltac:(lia)).
  destruct (apart (bind_variable r x n) news body) as [rest body'].
  cbn [snd] in IH |- *. cbn [bound_names firstn]. rewrite IH. reflexivity.
Qed.

Lemma handler_binds (params : list variable) (metadata : handler_metadata)
    (body : expr) :
  apart_binds body -> apart_handler_binds (Handler params metadata body).
Proof.
  intros IH r news Hcount. cbn [binders_in_handler] in Hcount |- *.
  cbn [apart_handler].
  rewrite (bind_variables_eq params r news ltac:(lia)). cbv beta iota.
  specialize
    (IH (snd (bind_variables r news params)) (skipn (length params) news)
       ltac:(rewrite length_skipn; lia)).
  destruct
    (apart (snd (bind_variables r news params)) (skipn (length params) news)
       body)
    as [rest body'].
  cbn [snd] in IH |- *. cbn [bound_names_in_handler].
  rewrite IH, firstn_plus. reflexivity.
Qed.

Lemma let_cont_binds (k : continuation) (h : continuation_handler)
    (body : expr) :
  apart_handler_binds h -> apart_binds body ->
  apart_binds (Let_cont_non_recursive k h body).
Proof.
  intros IHh IHbody r news Hcount. cbn [binders] in Hcount |- *.
  cbn [apart]. specialize (IHh r news ltac:(lia)).
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, IHh. subst rest. rewrite firstn_plus.
  assert (Hlength :
    S (binders body) <= length (skipn (binders_in_handler h) news))
    by (rewrite length_skipn; lia).
  destruct (skipn (binders_in_handler h) news) as [ | m rest];
    [ cbn [length] in Hlength; lia | ].
  cbn [length] in Hlength. cbn [new_continuation hd tl].
  specialize (IHbody (bind_continuation r k m) rest ltac:(lia)).
  destruct (apart (bind_continuation r k m) rest body) as [rest' body'].
  cbn [snd] in IHbody |- *. cbn [bound_names firstn].
  rewrite IHh, IHbody. reflexivity.
Qed.

Lemma thread_bindings_binds (r : renaming)
    (bindings : list static_binding) :
  Forall apart_binding_binds bindings ->
  ∀ news : list positive,
    list_sum (map binders_in_binding bindings) <= length news ->
    flat_map bound_names_in_binding
      (snd (thread (apart_binding r) news bindings))
    = firstn (list_sum (map binders_in_binding bindings)) news.
Proof.
  intros Hall.
  induction Hall as [ | binding bindings Hb _ IH]; intros news Hcount;
    [ reflexivity | ].
  cbn [map] in Hcount |- *. rewrite list_sum_cons in Hcount |- *.
  cbn [thread]. specialize (Hb r news ltac:(lia)).
  pose proof (apart_binding_rest binding r news) as Hrest.
  destruct (apart_binding r news binding) as [rest binding'].
  cbn [fst snd] in Hrest, Hb. subst rest.
  specialize
    (IH (skipn (binders_in_binding binding) news)
       ltac:(rewrite length_skipn; lia)).
  destruct
    (thread (apart_binding r) (skipn (binders_in_binding binding) news)
       bindings)
    as [rest' bindings'].
  cbn [snd] in IH |- *. cbn [flat_map]. rewrite Hb, IH, firstn_plus.
  reflexivity.
Qed.

Lemma let_static_binds (bindings : list static_binding) (body : expr) :
  Forall apart_binding_binds bindings -> apart_binds body ->
  apart_binds (Let_static bindings body).
Proof.
  intros IHbindings IHbody r news Hcount. cbn [binders] in Hcount |- *.
  cbn [apart].
  pose proof (thread_bindings_binds r bindings IHbindings news ltac:(lia))
    as Hbs.
  pose proof (thread_bindings_rest r bindings news) as Hrest.
  destruct (thread (apart_binding r) news bindings) as [rest bindings'].
  cbn [fst snd] in Hrest, Hbs. subst rest.
  specialize
    (IHbody r (skipn (list_sum (map binders_in_binding bindings)) news)
       ltac:(rewrite length_skipn; lia)).
  destruct
    (apart r (skipn (list_sum (map binders_in_binding bindings)) news) body)
    as [rest' body'].
  cbn [snd] in IHbody |- *. cbn [bound_names].
  rewrite Hbs, IHbody, firstn_plus. reflexivity.
Qed.

Lemma thread_handlers_binds (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun kh => apart_handler_binds (snd kh)) handlers ->
  ∀ news : list positive,
    list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
    <= length news ->
    flat_map bound_names_in_handler
      (snd
         (thread (fun news kh => apart_handler r news (snd kh)) news
            handlers))
    = firstn
        (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
        news.
Proof.
  intros Hall.
  induction Hall as [ | [k h] handlers Hh _ IH]; intros news Hcount;
    [ reflexivity | ].
  cbn [map snd] in Hcount, Hh |- *. rewrite list_sum_cons in Hcount |- *.
  cbn [thread snd]. specialize (Hh r news ltac:(lia)).
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, Hh. subst rest.
  specialize
    (IH (skipn (binders_in_handler h) news) ltac:(rewrite length_skipn; lia)).
  destruct
    (thread (fun news0 kh => apart_handler r news0 (snd kh))
       (skipn (binders_in_handler h) news) handlers)
    as [rest' handlers'].
  cbn [snd] in IH |- *. cbn [flat_map]. rewrite Hh, IH, firstn_plus.
  reflexivity.
Qed.

Lemma let_cont_recursive_binds
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun kh => apart_handler_binds (snd kh)) handlers ->
  apart_binds body ->
  apart_binds (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers IHbody r news Hcount. cbn [binders] in Hcount |- *.
  cbn [apart].
  assert (Hlength : length (map fst handlers) <= length news)
    by (rewrite length_map; lia).
  rewrite (bind_continuations_eq (map fst handlers) r news Hlength).
  cbv beta iota. rewrite length_map.
  set (r1 := snd (bind_continuations r news (map fst handlers))).
  pose proof
    (thread_handlers_binds r1 handlers IHhandlers
       (skipn (length handlers) news) ltac:(rewrite length_skipn; lia))
    as Hhs.
  pose proof
    (thread_length (fun news0 kh => apart_handler r1 news0 (snd kh)) handlers
       (skipn (length handlers) news))
    as Hlen.
  pose proof (thread_handlers_rest r1 handlers (skipn (length handlers) news))
    as Hrest.
  destruct
    (thread (fun news0 kh => apart_handler r1 news0 (snd kh))
       (skipn (length handlers) news) handlers)
    as [rest hs'].
  cbn [fst snd] in Hrest, Hhs, Hlen. subst rest.
  specialize
    (IHbody r1
       (skipn
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          (skipn (length handlers) news))
       ltac:(rewrite !length_skipn; lia)).
  destruct
    (apart r1
       (skipn
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          (skipn (length handlers) news))
       body)
    as [rest' body'].
  cbn [snd] in IHbody |- *. cbn [bound_names].
  assert (Hks : length (firstn (length handlers) news) = length hs')
    by (rewrite Hlen; apply firstn_length_le; lia).
  rewrite combine_fst by exact Hks.
  rewrite (flat_map_after bound_names_in_handler snd).
  rewrite combine_snd by exact Hks. rewrite Hhs, IHbody.
  rewrite <- Nat.add_assoc, !firstn_plus. reflexivity.
Qed.

Lemma code_binds (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr) :
  apart_binds body ->
  apart_code_binds (Code params my_closure ret exn metadata body).
Proof.
  intros IH r news Hcount. cbn [binders_in_code] in Hcount |- *.
  destruct news as [ | n news]; [ cbn [length] in Hcount; lia | ].
  cbn [length] in Hcount. cbn [apart_code new_variable hd tl].
  rewrite
    (bind_variables_eq params (bind_variable r my_closure n) news ltac:(lia)).
  cbv beta iota. cbn [firstn]. rewrite firstn_plus.
  assert (Hlength :
    S (S (binders body)) <= length (skipn (length params) news))
    by (rewrite length_skipn; lia).
  destruct (skipn (length params) news) as [ | m1 [ | m2 rest]];
    [ cbn [length] in Hlength; lia | cbn [length] in Hlength; lia | ].
  cbn [length] in Hlength. cbn [new_continuation hd tl].
  match goal with
  | |- context [apart ?r4 rest body] =>
    specialize (IH r4 rest ltac:(lia));
    destruct (apart r4 rest body) as [rest' body']
  end.
  cbn [snd] in IH |- *. cbn [bound_names_in_code firstn]. rewrite IH.
  reflexivity.
Qed.

Lemma bind_code_binds (id : code_id) (c : code) :
  apart_code_binds c -> apart_binding_binds (Bind_code id c).
Proof.
  intros IH r news Hcount. cbn [apart_binding].
  specialize (IH r news Hcount).
  destruct (apart_code r news c) as [rest c']. exact IH.
Qed.

Lemma apart_binds_all :
  (∀ e : expr, apart_binds e)
  ∧ (∀ h : continuation_handler, apart_handler_binds h)
  ∧ (∀ binding : static_binding, apart_binding_binds binding)
  ∧ (∀ c : code, apart_code_binds c).
Proof.
  apply term_mutual_ind.
  - exact let_binds.
  - exact let_static_binds.
  - exact let_cont_binds.
  - exact let_cont_recursive_binds.
  - intros ap r news _. reflexivity.
  - intros ac r news _. reflexivity.
  - intros scrutinee arms r news _. reflexivity.
  - intros r news _. reflexivity.
  - exact handler_binds.
  - exact bind_code_binds.
  - intros id r news _. reflexivity.
  - intros s const r news _. reflexivity.
  - exact code_binds.
Qed.

Theorem renamed_apart_avoiding_bound_names :
  renamed_apart_avoiding_bound_names_statement.
Proof.
  intros avoid e. unfold renamed_apart_avoiding.
  destruct (fresh_names_fresh (binders e) (avoid ++ names_of e))
    as [Hlength _].
  rewrite
    (proj1 apart_binds_all e identity_renaming _
       ltac:(rewrite Hlength; apply le_n)).
  apply firstn_all2. rewrite Hlength. apply le_n.
Qed.

Theorem canonical_avoiding_bound_names :
  canonical_avoiding_bound_names_statement.
Proof.
  intros avoid e. unfold canonical_avoiding.
  destruct
    (fresh_names_fresh (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    as [Hlength _].
  rewrite
    (proj1 apart_binds_all e identity_renaming _
       ltac:(rewrite Hlength; apply le_n)).
  apply firstn_all2. rewrite Hlength. apply le_n.
Qed.

Theorem renamed_apart_avoiding_binders_apart :
  renamed_apart_avoiding_binders_apart_statement.
Proof.
  intros avoid e. unfold binders_apart.
  rewrite renamed_apart_avoiding_bound_names.
  destruct (fresh_names_fresh (binders e) (avoid ++ names_of e))
    as [_ [Hdistinct Houtside]].
  rewrite Hdistinct. cbn [andb]. apply forallb_forall. intros name Hname.
  rewrite (Houtside name (proj2 (among_In _ _) Hname)). reflexivity.
Qed.

Theorem canonical_avoiding_binders_apart :
  canonical_avoiding_binders_apart_statement.
Proof.
  intros avoid e. unfold binders_apart.
  rewrite canonical_avoiding_bound_names.
  destruct
    (fresh_names_fresh (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    as [_ [Hdistinct Houtside]].
  rewrite Hdistinct. cbn [andb]. apply forallb_forall. intros name Hname.
  rewrite (Houtside name (proj2 (among_In _ _) Hname)). reflexivity.
Qed.

(* ======================================================================== *)
(* A name of a term is free in it or bound in it                            *)
(* ======================================================================== *)

Definition names_split (e : expr) : Prop :=
  (∀ x : variable,
    In x (variables_of e) ->
    In x (free_variables_of_expr e) ∨ In x (bound_names e))
  ∧ (∀ k : continuation,
      In k (continuations_of e) ->
      In k (free_continuations_of_expr e) ∨ In k (bound_names e)).

Definition names_in_handler_split (h : continuation_handler) : Prop :=
  (∀ x : variable,
    In x (variables_in_handler h) ->
    In x (free_variables_of_handler h) ∨ In x (bound_names_in_handler h))
  ∧ (∀ k : continuation,
      In k (continuations_in_handler h) ->
      In k (free_continuations_of_handler h)
      ∨ In k (bound_names_in_handler h)).

Definition names_in_binding_split (binding : static_binding) : Prop :=
  (∀ x : variable,
    In x (variables_in_binding binding) ->
    In x (free_variables_of_binding binding)
    ∨ In x (bound_names_in_binding binding))
  ∧ (∀ k : continuation,
      In k (continuations_in_binding binding) ->
      In k (free_continuations_of_binding binding)
      ∨ In k (bound_names_in_binding binding)).

Definition names_in_code_split (c : code) : Prop :=
  (∀ x : variable,
    In x (variables_in_code c) ->
    In x (free_variables_of_code c) ∨ In x (bound_names_in_code c))
  ∧ (∀ k : continuation,
      In k (continuations_in_code c) ->
      In k (free_continuations_of_code c) ∨ In k (bound_names_in_code c)).

Lemma let_split (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  names_split body -> names_split (Let x kind d body).
Proof.
  intros [IHv IHk]. split.
  - intros y Hy. cbn [variables_of] in Hy.
    cbn [free_variables_of_expr bound_names].
    destruct Hy as [Heq | Hy]; [ right; left; exact Heq | ].
    apply in_app_or in Hy as [Hy | Hy].
    + left. apply in_or_app. left.
      rewrite free_variables_of_named_variables. exact Hy.
    + destruct (IHv y Hy) as [Hfree | Hbound];
        [ | right; right; exact Hbound ].
      destruct (in_without_one y x _ Hfree) as [Heq | Hin];
        [ right; left; symmetry; exact Heq
        | left; apply in_or_app; right; exact Hin ].
  - intros k Hk. cbn [continuations_of] in Hk.
    cbn [free_continuations_of_expr bound_names].
    destruct (IHk k Hk) as [Hfree | Hbound];
      [ left; exact Hfree | right; right; exact Hbound ].
Qed.

Lemma handler_split (params : list variable) (metadata : handler_metadata)
    (body : expr) :
  names_split body -> names_in_handler_split (Handler params metadata body).
Proof.
  intros [IHv IHk]. split.
  - intros y Hy. cbn [variables_in_handler] in Hy.
    cbn [free_variables_of_handler bound_names_in_handler].
    apply in_app_or in Hy as [Hy | Hy];
      [ right; apply in_or_app; left; exact Hy | ].
    destruct (IHv y Hy) as [Hfree | Hbound];
      [ | right; apply in_or_app; right; exact Hbound ].
    destruct (in_without y params _ Hfree) as [Hin | Hin];
      [ right; apply in_or_app; left; exact Hin | left; exact Hin ].
  - intros k Hk. cbn [continuations_in_handler] in Hk.
    cbn [free_continuations_of_handler bound_names_in_handler].
    destruct (IHk k Hk) as [Hfree | Hbound];
      [ left; exact Hfree | right; apply in_or_app; right; exact Hbound ].
Qed.

Lemma let_cont_split (k : continuation) (h : continuation_handler)
    (body : expr) :
  names_in_handler_split h -> names_split body ->
  names_split (Let_cont_non_recursive k h body).
Proof.
  intros [IHhv IHhk] [IHv IHk]. split.
  - intros y Hy. cbn [variables_of] in Hy.
    cbn [free_variables_of_expr bound_names].
    apply in_app_or in Hy as [Hy | Hy].
    + destruct (IHhv y Hy) as [Hfree | Hbound];
        [ left; apply in_or_app; left; exact Hfree
        | right; apply in_or_app; left; exact Hbound ].
    + destruct (IHv y Hy) as [Hfree | Hbound];
        [ left; apply in_or_app; right; exact Hfree
        | right; apply in_or_app; right; right; exact Hbound ].
  - intros j Hj. cbn [continuations_of] in Hj.
    cbn [free_continuations_of_expr bound_names].
    destruct Hj as [Heq | Hj];
      [ right; apply in_or_app; right; left; exact Heq | ].
    apply in_app_or in Hj as [Hj | Hj].
    + destruct (IHhk j Hj) as [Hfree | Hbound];
        [ left; apply in_or_app; left; exact Hfree
        | right; apply in_or_app; left; exact Hbound ].
    + destruct (IHk j Hj) as [Hfree | Hbound];
        [ | right; apply in_or_app; right; right; exact Hbound ].
      destruct (in_without_one j k _ Hfree) as [Heq | Hin];
        [ right; apply in_or_app; right; left; symmetry; exact Heq
        | left; apply in_or_app; right; exact Hin ].
Qed.

Lemma let_static_split (bindings : list static_binding) (body : expr) :
  Forall names_in_binding_split bindings -> names_split body ->
  names_split (Let_static bindings body).
Proof.
  intros IHbindings [IHv IHk]. rewrite Forall_forall in IHbindings. split.
  - intros y Hy. cbn [variables_of] in Hy.
    cbn [free_variables_of_expr bound_names].
    apply in_app_or in Hy as [Hy | Hy].
    + apply in_flat_map in Hy as (binding & Hin & Hy).
      destruct (proj1 (IHbindings binding Hin) y Hy) as [Hfree | Hbound];
        [ left | right ]; apply in_or_app; left; apply in_flat_map;
        exists binding; (split; [ exact Hin | assumption ]).
    + destruct (IHv y Hy) as [Hfree | Hbound]; [ left | right ];
        apply in_or_app; right; assumption.
  - intros j Hj. cbn [continuations_of] in Hj.
    cbn [free_continuations_of_expr bound_names].
    apply in_app_or in Hj as [Hj | Hj].
    + apply in_flat_map in Hj as (binding & Hin & Hj).
      destruct (proj2 (IHbindings binding Hin) j Hj) as [Hfree | Hbound];
        [ left | right ]; apply in_or_app; left; apply in_flat_map;
        exists binding; (split; [ exact Hin | assumption ]).
    + destruct (IHk j Hj) as [Hfree | Hbound]; [ left | right ];
        apply in_or_app; right; assumption.
Qed.

Lemma let_cont_recursive_split
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun kh => names_in_handler_split (snd kh)) handlers ->
  names_split body ->
  names_split (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers [IHv IHk]. rewrite Forall_forall in IHhandlers. split.
  - intros y Hy. cbn [variables_of] in Hy.
    cbn [free_variables_of_expr bound_names].
    apply in_app_or in Hy as [Hy | Hy].
    + apply in_flat_map in Hy as ([k h] & Hin & Hy). cbn [snd] in Hy.
      destruct (proj1 (IHhandlers (k, h) Hin) y Hy) as [Hfree | Hbound].
      * left. apply in_or_app. left. apply in_flat_map. exists (k, h).
        split; [ exact Hin | exact Hfree ].
      * right. apply in_or_app. right. apply in_or_app. left.
        apply in_flat_map. exists (k, h).
        split; [ exact Hin | exact Hbound ].
    + destruct (IHv y Hy) as [Hfree | Hbound];
        [ left; apply in_or_app; right; exact Hfree
        | right; apply in_or_app; right; apply in_or_app; right;
          exact Hbound ].
  - intros j Hj. cbn [continuations_of] in Hj.
    cbn [free_continuations_of_expr bound_names].
    assert (Hcases :
      In j (map fst handlers)
      ∨ In j
          (flat_map (fun '(_, h) => free_continuations_of_handler h)
             handlers
           ++ free_continuations_of_expr body)
      ∨ In j
          (flat_map (fun kh => bound_names_in_handler (snd kh)) handlers
           ++ bound_names body)).
    { apply in_app_or in Hj as [Hj | Hj].
      - apply in_flat_map in Hj as ([k h] & Hin & Hj). cbn [fst snd] in Hj.
        destruct Hj as [Heq | Hj].
        + left. apply in_map_iff. exists (k, h).
          split; [ exact Heq | exact Hin ].
        + destruct (proj2 (IHhandlers (k, h) Hin) j Hj) as [Hfree | Hbound].
          * right. left. apply in_or_app. left. apply in_flat_map.
            exists (k, h). split; [ exact Hin | exact Hfree ].
          * right. right. apply in_or_app. left. apply in_flat_map.
            exists (k, h). split; [ exact Hin | exact Hbound ].
      - destruct (IHk j Hj) as [Hfree | Hbound];
          [ right; left; apply in_or_app; right; exact Hfree
          | right; right; apply in_or_app; right; exact Hbound ]. }
    destruct Hcases as [Hin | [Hfree | Hbound]].
    + right. apply in_or_app. left. exact Hin.
    + destruct (in_without j (map fst handlers) _ Hfree) as [Hin | Hin];
        [ right; apply in_or_app; left; exact Hin | left; exact Hin ].
    + right. apply in_or_app. right. exact Hbound.
Qed.

Lemma code_split (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr) :
  names_split body ->
  names_in_code_split (Code params my_closure ret exn metadata body).
Proof.
  intros [IHv IHk]. split.
  - intros y Hy. cbn [variables_in_code] in Hy.
    cbn [free_variables_of_code bound_names_in_code].
    destruct Hy as [Heq | Hy]; [ right; left; exact Heq | ].
    apply in_app_or in Hy as [Hy | Hy];
      [ right; right; apply in_or_app; left; exact Hy | ].
    destruct (IHv y Hy) as [Hfree | Hbound];
      [ | right; right; apply in_or_app; right; right; right; exact Hbound ].
    destruct (in_without y (my_closure :: params) _ Hfree)
      as [[Heq | Hin] | Hin];
      [ right; left; exact Heq
      | right; right; apply in_or_app; left; exact Hin | left; exact Hin ].
  - intros j Hj. cbn [continuations_in_code] in Hj.
    cbn [free_continuations_of_code bound_names_in_code].
    destruct Hj as [Heq | [Heq | Hj]];
      [ right; right; apply in_or_app; right; left; exact Heq
      | right; right; apply in_or_app; right; right; left; exact Heq | ].
    destruct (IHk j Hj) as [Hfree | Hbound];
      [ | right; right; apply in_or_app; right; right; right; exact Hbound ].
    destruct (in_without j [ret; exn] _ Hfree) as [[Heq | [Heq | []]] | Hin];
      [ right; right; apply in_or_app; right; left; exact Heq
      | right; right; apply in_or_app; right; right; left; exact Heq
      | left; exact Hin ].
Qed.

Lemma names_split_all :
  (∀ e : expr, names_split e)
  ∧ (∀ h : continuation_handler, names_in_handler_split h)
  ∧ (∀ binding : static_binding, names_in_binding_split binding)
  ∧ (∀ c : code, names_in_code_split c).
Proof.
  apply term_mutual_ind.
  - exact let_split.
  - exact let_static_split.
  - exact let_cont_split.
  - exact let_cont_recursive_split.
  - intros ap. split; intros a Ha; left; exact Ha.
  - intros ac. split; intros a Ha; left; exact Ha.
  - intros scrutinee arms. split; intros a Ha; left; exact Ha.
  - split; intros a [].
  - exact handler_split.
  - intros id c IH. exact IH.
  - intros id. split; intros a [].
  - intros s const.
    split; [ intros a Ha; left; destruct const; exact Ha | intros a [] ].
  - exact code_split.
Qed.

(* ======================================================================== *)
(* The canonical form is the term up to bound names                         *)
(* ======================================================================== *)

(* The term is first renamed apart with new names above those of the
   canonical form too: the names that the result binds are then outside
   the new names of the canonical form, and its free names are those of
   the term, so the new names of the canonical form are outside every
   name of the result. The canonical form of the result is that of the
   term. *)
Theorem canonical_avoiding_alpha : canonical_avoiding_alpha_statement.
Proof.
  intros avoid e Hclosed Hdistinct.
  set
    (taken :=
       fresh_names (binders e)
         (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e)).
  set (d := renamed_apart_avoiding taken e).
  assert (Hed : alpha_equivalent e d)
    by exact (renamed_apart_avoiding_alpha taken e Hclosed Hdistinct).
  rewrite (alpha_equivalent_canonical avoid e d Hed).
  apply (Alpha_trans e d _ Hed). unfold canonical_avoiding.
  rewrite <- (canonical_supply_alpha avoid e d Hed). fold taken.
  pose proof (proj1 apart_alpha_all d identity_renaming taken) as Halpha.
  rewrite (rename_expr_identity d) in Halpha. apply Halpha.
  - rewrite <- (alpha_equivalent_code_closed e d Hed). exact Hclosed.
  - rewrite <- (alpha_equivalent_binders_distinct e d Hed). exact Hdistinct.
  - destruct
      (fresh_names_fresh (binders e)
         (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
      as [Hlength [Hnodup Houtside]].
    fold taken in Hlength, Hnodup, Houtside.
    split; [ apply distinct_NoDup, Hnodup | split ].
    + intros a Ha.
      assert (Hnot_bound : ~ In a (bound_names d)).
      { unfold d. rewrite renamed_apart_avoiding_bound_names. intros Hin.
        destruct (fresh_names_fresh (binders e) (taken ++ names_of e))
          as [_ [_ Houtside']].
        pose proof (Houtside' a (proj2 (among_In _ _) Hin)) as Hnot.
        rewrite among_app, (proj2 (among_In a taken) Ha) in Hnot.
        discriminate Hnot. }
      pose proof (Houtside a (proj2 (among_In _ _) Ha)) as Hnot.
      rewrite !among_app in Hnot.
      apply orb_false_iff in Hnot as [_ Hnot].
      apply orb_false_iff in Hnot as [Hv Hk].
      split; intros Hin.
      * destruct (proj1 (proj1 names_split_all d) a Hin) as [Hfree | Hbound];
          [ | exact (Hnot_bound Hbound) ].
        apply among_In in Hfree.
        rewrite <- (alpha_equivalent_free_variables e d Hed a) in Hfree.
        rewrite Hfree in Hv. discriminate Hv.
      * destruct (proj2 (proj1 names_split_all d) a Hin) as [Hfree | Hbound];
          [ | exact (Hnot_bound Hbound) ].
        apply among_In in Hfree.
        rewrite <- (alpha_equivalent_free_continuations e d Hed a) in Hfree.
        rewrite Hfree in Hk. discriminate Hk.
    + rewrite <- (alpha_equivalent_binders e d Hed), Hlength. apply le_n.
Qed.

Theorem alpha_equivalent_iff_canonical :
  alpha_equivalent_iff_canonical_statement.
Proof.
  intros avoid e e' Hclosed Hdistinct Hclosed' Hdistinct'. split.
  - apply alpha_equivalent_canonical.
  - intros Hsame.
    apply
      (Alpha_trans e (canonical_avoiding avoid e) e'
         (canonical_avoiding_alpha avoid e Hclosed Hdistinct)).
    rewrite Hsame. apply Alpha_sym.
    exact (canonical_avoiding_alpha avoid e' Hclosed' Hdistinct').
Qed.

Theorem canonical_avoiding_idempotent :
  canonical_avoiding_idempotent_statement.
Proof.
  intros avoid e Hclosed Hdistinct. symmetry.
  exact
    (alpha_equivalent_canonical avoid e (canonical_avoiding avoid e)
       (canonical_avoiding_alpha avoid e Hclosed Hdistinct)).
Qed.

Theorem canonical_avoiding_equivalent :
  canonical_avoiding_equivalent_statement.
Proof.
  intros avoid e Hclosed Hdistinct.
  exact
    (alpha_equivalent_sound e (canonical_avoiding avoid e)
       (canonical_avoiding_alpha avoid e Hclosed Hdistinct)).
Qed.

Theorem canonical_avoiding_accepted : canonical_avoiding_accepted_statement.
Proof.
  intros avoid e Hclosed Hdistinct.
  exact
    (alpha_equivalent_accepted e (canonical_avoiding avoid e)
       (canonical_avoiding_alpha avoid e Hclosed Hdistinct)).
Qed.

(* ======================================================================== *)
(* The free names of the candidate                                          *)
(* ======================================================================== *)

Lemma in_without_intro (name : positive) (bound names : list positive) :
  In name names -> ~ In name bound -> In name (without bound names).
Proof.
  intros Hin Hnot.
  apply among_In, among_without_all; [ exact Hnot | apply among_In, Hin ].
Qed.

Lemma in_without_elim (name : positive) (bound names : list positive) :
  In name (without bound names) -> In name names ∧ ~ In name bound.
Proof. intros Hin. apply among_without_in, among_In, Hin. Qed.

Lemma bind_variable_other (r : renaming) (x z : variable) (n : positive) :
  z <> x -> rename_variable r z <> n ->
  rename_variable (bind_variable r x n) z = rename_variable r z.
Proof.
  intros Hne Hnot. rewrite rename_variable_bind_variable.
  apply swap_other; [ | exact Hnot ].
  intros Heq. apply Hne. exact (rename_variable_injective r _ _ Heq).
Qed.

Lemma bind_continuation_other (r : renaming) (k z : continuation)
    (n : positive) :
  z <> k -> rename_continuation r z <> n ->
  rename_continuation (bind_continuation r k n) z = rename_continuation r z.
Proof.
  intros Hne Hnot. rewrite rename_continuation_bind_continuation.
  apply swap_other; [ | exact Hnot ].
  intros Heq. apply Hne. exact (rename_continuation_injective r _ _ Heq).
Qed.

(* What a bound name becomes is one of the new names taken. *)
Lemma bind_variables_bound_in (xs : list variable) :
  ∀ (r : renaming) (news : list positive) (y : variable),
    NoDup news ->
    length xs <= length news ->
    In y xs ->
    In (rename_variable (snd (bind_variables r news xs)) y)
      (firstn (length xs) news).
Proof.
  induction xs as [ | x xs IH]; intros r news y Hnodup Hlength Hy;
    [ destruct Hy | ].
  destruct news as [ | n news]; [ cbn [length] in Hlength; lia | ].
  cbn [length] in Hlength.
  cbn [bind_variables new_variable hd tl length firstn].
  pose proof (IH (bind_variable r x n) news y) as IH'.
  pose proof (bind_variables_unbound xs (bind_variable r x n) news y)
    as Hunbound.
  destruct (bind_variables (bind_variable r x n) news xs)
    as [[rest xs'] r'].
  cbn [snd] in IH', Hunbound |- *.
  assert (Hnot : ~ In n news) by (inversion Hnodup; assumption).
  destruct (in_dec Pos.eq_dec y xs) as [Hin | Hnotin].
  - right. apply IH'; [ inversion Hnodup; assumption | lia | exact Hin ].
  - destruct Hy as [Heq | Hin]; [ subst y | contradiction ].
    left. rewrite Hunbound; [ | lia | exact Hnotin | ].
    + rewrite rename_variable_bind_variable, swap_first. reflexivity.
    + rewrite rename_variable_bind_variable, swap_first. intros Hin.
      exact (Hnot (in_firstn _ _ _ Hin)).
Qed.

Lemma bind_continuations_bound_in (ks : list continuation) :
  ∀ (r : renaming) (news : list positive) (j : continuation),
    NoDup news ->
    length ks <= length news ->
    In j ks ->
    In (rename_continuation (snd (bind_continuations r news ks)) j)
      (firstn (length ks) news).
Proof.
  induction ks as [ | k ks IH]; intros r news j Hnodup Hlength Hj;
    [ destruct Hj | ].
  destruct news as [ | n news]; [ cbn [length] in Hlength; lia | ].
  cbn [length] in Hlength.
  cbn [bind_continuations new_continuation hd tl length firstn].
  pose proof (IH (bind_continuation r k n) news j) as IH'.
  pose proof (bind_continuations_unbound ks (bind_continuation r k n) news j)
    as Hunbound.
  destruct (bind_continuations (bind_continuation r k n) news ks)
    as [[rest ks'] r'].
  cbn [snd] in IH', Hunbound |- *.
  assert (Hnot : ~ In n news) by (inversion Hnodup; assumption).
  destruct (in_dec Pos.eq_dec j ks) as [Hin | Hnotin].
  - right. apply IH'; [ inversion Hnodup; assumption | lia | exact Hin ].
  - destruct Hj as [Heq | Hin]; [ subst j | contradiction ].
    left. rewrite Hunbound; [ | lia | exact Hnotin | ].
    + rewrite rename_continuation_bind_continuation, swap_first.
      reflexivity.
    + rewrite rename_continuation_bind_continuation, swap_first.
      intros Hin. exact (Hnot (in_firstn _ _ _ Hin)).
Qed.

(* Binding variables leaves the continuations alone, and the converse. *)
Lemma bind_variables_continuation (xs : list variable) (r : renaming)
    (news : list positive) (k : continuation) :
  rename_continuation (snd (bind_variables r news xs)) k
  = rename_continuation r k.
Proof.
  unfold rename_continuation. rewrite bind_variables_continuations.
  reflexivity.
Qed.

Lemma bind_continuations_variable (ks : list continuation) (r : renaming)
    (news : list positive) (x : variable) :
  rename_variable (snd (bind_continuations r news ks)) x
  = rename_variable r x.
Proof.
  unfold rename_variable. rewrite bind_continuations_variables.
  reflexivity.
Qed.

(* The free names of the candidate are among what the renaming makes of
   the free names of the term. *)
Definition apart_free (e : expr) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied r news e ->
    incl (free_variables_of_expr (snd (apart r news e)))
      (map (rename_variable r) (free_variables_of_expr e))
    ∧ incl (free_continuations_of_expr (snd (apart r news e)))
        (map (rename_continuation r) (free_continuations_of_expr e)).

Definition apart_handler_free (h : continuation_handler) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied_names r news (binders_in_handler h)
      (free_variables_of_handler h) (free_continuations_of_handler h) ->
    incl (free_variables_of_handler (snd (apart_handler r news h)))
      (map (rename_variable r) (free_variables_of_handler h))
    ∧ incl (free_continuations_of_handler (snd (apart_handler r news h)))
        (map (rename_continuation r) (free_continuations_of_handler h)).

Definition apart_binding_free (binding : static_binding) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied_names r news (binders_in_binding binding)
      (free_variables_of_binding binding)
      (free_continuations_of_binding binding) ->
    incl (free_variables_of_binding (snd (apart_binding r news binding)))
      (map (rename_variable r) (free_variables_of_binding binding))
    ∧ incl
        (free_continuations_of_binding (snd (apart_binding r news binding)))
        (map (rename_continuation r)
           (free_continuations_of_binding binding)).

Definition apart_code_free (c : code) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied_names r news (binders_in_code c) (free_variables_of_code c)
      (free_continuations_of_code c) ->
    incl (free_variables_of_code (snd (apart_code r news c)))
      (map (rename_variable r) (free_variables_of_code c))
    ∧ incl (free_continuations_of_code (snd (apart_code r news c)))
        (map (rename_continuation r) (free_continuations_of_code c)).

Lemma renamed_free (r : renaming) (news : list positive) (e : expr) :
  snd (apart r news e) = rename_expr r e ->
  incl (free_variables_of_expr (snd (apart r news e)))
    (map (rename_variable r) (free_variables_of_expr e))
  ∧ incl (free_continuations_of_expr (snd (apart r news e)))
      (map (rename_continuation r) (free_continuations_of_expr e)).
Proof.
  intros ->.
  rewrite free_variables_of_expr_renamed, free_continuations_of_expr_renamed.
  split; apply incl_refl.
Qed.

Lemma let_free (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  apart_free body -> apart_free (Let x kind d body).
Proof.
  intros IH r news Hsupplied.
  pose proof Hsupplied as (Hnodup & Hcount & [Hov Hok]).
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hcount, Hov, Hok.
  destruct news as [ | n news]; [ cbn [length] in Hcount; lia | ].
  assert (Hbody : supplied (bind_variable r x n) news body).
  { apply (supplied_bind_variable x r n news _ _ _ _ _ Hsupplied (le_n _)).
    intros y Hy.
    destruct (in_without_one y x _ Hy) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ]. }
  destruct (IH (bind_variable r x n) news Hbody) as [IHv IHk].
  cbn [apart new_variable hd tl].
  destruct (apart (bind_variable r x n) news body) as [rest body'].
  cbn [snd] in IHv, IHk |- *.
  cbn [free_variables_of_expr free_continuations_of_expr]. split.
  - intros y Hy. rewrite map_app. apply in_or_app.
    apply in_app_or in Hy as [Hy | Hy].
    + left. rewrite free_variables_of_named_renamed in Hy. exact Hy.
    + right. apply in_without_elim in Hy as [Hy Hne].
      apply IHv, in_map_iff in Hy as (z & <- & Hz).
      destruct (Pos.eq_dec z x) as [-> | Hzx].
      * exfalso. apply Hne. left.
        rewrite rename_variable_bind_variable, swap_first. reflexivity.
      * assert (Hfree : In z (without [x] (free_variables_of_expr body)))
          by (apply in_without_intro; [ exact Hz | ];
              intros [Heq | []]; apply Hzx; symmetry; exact Heq).
        rewrite (bind_variable_other r x z n Hzx);
          [ apply in_map; exact Hfree | ].
        intros Heq. apply (Hov z); [ apply in_or_app; right; exact Hfree | ].
        left. symmetry. exact Heq.
  - exact IHk.
Qed.

Lemma handler_free (params : list variable) (metadata : handler_metadata)
    (body : expr) :
  apart_free body -> apart_handler_free (Handler params metadata body).
Proof.
  intros IH r news Hsupplied.
  pose proof Hsupplied as (Hnodup & Hcount & [Hov Hok]).
  cbn [binders_in_handler free_variables_of_handler
       free_continuations_of_handler]
    in Hcount, Hov, Hok.
  assert (Hlength : length params <= length news) by lia.
  assert (Hbody :
    supplied (snd (bind_variables r news params))
      (skipn (length params) news) body).
  { apply
      (supplied_bind_variables params r news _ _ _ _ _ Hsupplied (le_n _)).
    intros y Hy. exact (in_without y params _ Hy). }
  destruct (IH _ _ Hbody) as [IHv IHk].
  cbn [apart_handler].
  rewrite (bind_variables_eq params r news Hlength). cbv beta iota.
  destruct
    (apart (snd (bind_variables r news params)) (skipn (length params) news)
       body)
    as [rest body'].
  cbn [snd] in IHv, IHk |- *.
  cbn [free_variables_of_handler free_continuations_of_handler]. split.
  - intros y Hy. apply in_without_elim in Hy as [Hy Hne].
    apply IHv, in_map_iff in Hy as (z & <- & Hz).
    destruct (in_dec Pos.eq_dec z params) as [Hin | Hnotin].
    + exfalso. apply Hne.
      exact (bind_variables_bound_in params r news z Hnodup Hlength Hin).
    + assert (Hfree : In z (without params (free_variables_of_expr body)))
        by (apply in_without_intro; [ exact Hz | exact Hnotin ]).
      rewrite (bind_variables_unbound params r news z Hlength Hnotin);
        [ apply in_map; exact Hfree | ].
      intros Hin. exact (Hov z Hfree (in_firstn _ _ _ Hin)).
  - intros k Hk. apply IHk in Hk.
    rewrite (map_ext _ _ (bind_variables_continuation params r news)) in Hk.
    exact Hk.
Qed.

Lemma let_cont_free (k : continuation) (h : continuation_handler)
    (body : expr) :
  apart_handler_free h -> apart_free body ->
  apart_free (Let_cont_non_recursive k h body).
Proof.
  intros IHh IHbody r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  destruct
    (IHh r news
       (supplied_skipn 0 r news _ (binders_in_handler h) _
          (free_variables_of_handler h) _ (free_continuations_of_handler h)
          Hsupplied ltac:(lia) ltac:(apply incl_appl, incl_refl)
          ltac:(apply incl_appl, incl_refl)))
    as [IHhv IHhk].
  cbn [apart].
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, IHhv, IHhk. subst rest.
  pose proof
    (supplied_skipn (binders_in_handler h) r news _
       (S (binders body)) _ (free_variables_of_expr body) _
       (free_continuations_of_handler h
        ++ without [k] (free_continuations_of_expr body))
       Hsupplied (le_n _) ltac:(apply incl_appr, incl_refl)
       ltac:(apply incl_refl))
    as H1.
  destruct (skipn (binders_in_handler h) news) as [ | m rest];
    [ destruct H1 as (_ & Hcount & _); cbn [length] in Hcount; lia | ].
  cbn [new_continuation hd tl].
  assert (Hbody : supplied (bind_continuation r k m) rest body).
  { apply (supplied_bind_continuation k r m rest _ _ _ _ _ H1 (le_n _)).
    intros j Hj.
    destruct (in_without_one j k _ Hj) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ]. }
  destruct (IHbody _ _ Hbody) as [IHv IHk].
  destruct H1 as (_ & _ & [_ Hok]).
  destruct (apart (bind_continuation r k m) rest body) as [rest' body'].
  cbn [snd] in IHv, IHk |- *.
  cbn [free_variables_of_expr free_continuations_of_expr]. split.
  - intros y Hy. rewrite map_app. apply in_or_app.
    apply in_app_or in Hy as [Hy | Hy];
      [ left; exact (IHhv y Hy) | right; exact (IHv y Hy) ].
  - intros j Hj. rewrite map_app. apply in_or_app.
    apply in_app_or in Hj as [Hj | Hj]; [ left; exact (IHhk j Hj) | ].
    right. apply in_without_elim in Hj as [Hj Hne].
    apply IHk, in_map_iff in Hj as (z & <- & Hz).
    destruct (Pos.eq_dec z k) as [-> | Hzk].
    + exfalso. apply Hne. left.
      rewrite rename_continuation_bind_continuation, swap_first.
      reflexivity.
    + assert (Hfree : In z (without [k] (free_continuations_of_expr body)))
        by (apply in_without_intro; [ exact Hz | ];
            intros [Heq | []]; apply Hzk; symmetry; exact Heq).
      rewrite (bind_continuation_other r k z m Hzk);
        [ apply in_map; exact Hfree | ].
      intros Heq. apply (Hok z); [ apply in_or_app; right; exact Hfree | ].
      left. symmetry. exact Heq.
Qed.

Lemma thread_bindings_free (r : renaming) (bindings : list static_binding) :
  Forall apart_binding_free bindings ->
  ∀ (news : list positive) (vs : list variable) (ks : list continuation),
    supplied_names r news (list_sum (map binders_in_binding bindings)) vs ks
    ->
    (∀ binding : static_binding,
      In binding bindings ->
      incl (free_variables_of_binding binding) vs
      ∧ incl (free_continuations_of_binding binding) ks) ->
    incl
      (flat_map free_variables_of_binding
         (snd (thread (apart_binding r) news bindings)))
      (map (rename_variable r)
         (flat_map free_variables_of_binding bindings))
    ∧ incl
        (flat_map free_continuations_of_binding
           (snd (thread (apart_binding r) news bindings)))
        (map (rename_continuation r)
           (flat_map free_continuations_of_binding bindings)).
Proof.
  intros Hall.
  induction Hall as [ | binding bindings Hb _ IH];
    intros news vs ks Hsupplied Hincl;
    [ split; intros a [] | ].
  cbn [map] in Hsupplied. rewrite list_sum_cons in Hsupplied.
  destruct (Hincl binding (or_introl eq_refl)) as [Hvs Hks].
  destruct
    (Hb r news
       (supplied_skipn 0 r news _ (binders_in_binding binding) _ _ _ _
          Hsupplied ltac:(lia) Hvs Hks))
    as [Hbv Hbk].
  cbn [thread].
  pose proof (apart_binding_rest binding r news) as Hrest.
  destruct (apart_binding r news binding) as [rest binding'].
  cbn [fst snd] in Hrest, Hbv, Hbk. subst rest.
  destruct
    (IH (skipn (binders_in_binding binding) news) vs ks
       (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _)
          (incl_refl _) (incl_refl _))
       (fun b Hin => Hincl b (or_intror Hin)))
    as [IHv IHk].
  destruct
    (thread (apart_binding r) (skipn (binders_in_binding binding) news)
       bindings)
    as [rest' bindings'].
  cbn [snd] in IHv, IHk |- *. cbn [flat_map]. rewrite !map_app.
  split; apply incl_app;
    first [ apply incl_appl; assumption | apply incl_appr; assumption ].
Qed.

Lemma let_static_free (bindings : list static_binding) (body : expr) :
  Forall apart_binding_free bindings -> apart_free body ->
  apart_free (Let_static bindings body).
Proof.
  intros IHbindings IHbody r news Hsupplied. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  destruct
    (thread_bindings_free r bindings IHbindings news
       (flat_map free_variables_of_binding bindings
        ++ free_variables_of_expr body)
       (flat_map free_continuations_of_binding bindings
        ++ free_continuations_of_expr body)
       (supplied_skipn 0 r news _ (list_sum (map binders_in_binding bindings))
          _ _ _ _ Hsupplied ltac:(lia) (incl_refl _) (incl_refl _)))
    as [Hbv Hbk].
  { intros binding Hin. split; intros a Ha; apply in_or_app; left;
      apply in_flat_map; exists binding; (split; [ exact Hin | exact Ha ]). }
  cbn [apart].
  pose proof (thread_bindings_rest r bindings news) as Hrest.
  destruct (thread (apart_binding r) news bindings) as [rest bindings'].
  cbn [fst snd] in Hrest, Hbv, Hbk. subst rest.
  destruct
    (IHbody r (skipn (list_sum (map binders_in_binding bindings)) news)
       (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _)
          ltac:(apply incl_appr, incl_refl)
          ltac:(apply incl_appr, incl_refl)))
    as [IHv IHk].
  destruct
    (apart r (skipn (list_sum (map binders_in_binding bindings)) news) body)
    as [rest' body'].
  cbn [snd] in IHv, IHk |- *.
  cbn [free_variables_of_expr free_continuations_of_expr]. rewrite !map_app.
  split; apply incl_app;
    first [ apply incl_appl; assumption | apply incl_appr; assumption ].
Qed.

Lemma thread_handlers_free (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun kh => apart_handler_free (snd kh)) handlers ->
  ∀ (news : list positive) (vs : list variable) (ks : list continuation),
    supplied_names r news
      (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)) vs
      ks ->
    (∀ kh : continuation * continuation_handler,
      In kh handlers ->
      incl (free_variables_of_handler (snd kh)) vs
      ∧ incl (free_continuations_of_handler (snd kh)) ks) ->
    incl
      (flat_map free_variables_of_handler
         (snd
            (thread (fun news kh => apart_handler r news (snd kh)) news
               handlers)))
      (map (rename_variable r)
         (flat_map (fun kh => free_variables_of_handler (snd kh)) handlers))
    ∧ incl
        (flat_map free_continuations_of_handler
           (snd
              (thread (fun news kh => apart_handler r news (snd kh)) news
                 handlers)))
        (map (rename_continuation r)
           (flat_map (fun kh => free_continuations_of_handler (snd kh))
              handlers)).
Proof.
  intros Hall.
  induction Hall as [ | [k h] handlers Hh _ IH];
    intros news vs ks Hsupplied Hincl;
    [ split; intros a [] | ].
  cbn [map snd] in Hsupplied, Hh. rewrite list_sum_cons in Hsupplied.
  destruct (Hincl (k, h) (or_introl eq_refl)) as [Hvs Hks].
  cbn [snd] in Hvs, Hks.
  destruct
    (Hh r news
       (supplied_skipn 0 r news _ (binders_in_handler h) _ _ _ _ Hsupplied
          ltac:(lia) Hvs Hks))
    as [Hhv Hhk].
  cbn [thread snd].
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, Hhv, Hhk. subst rest.
  destruct
    (IH (skipn (binders_in_handler h) news) vs ks
       (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _)
          (incl_refl _) (incl_refl _))
       (fun kh Hin => Hincl kh (or_intror Hin)))
    as [IHv IHk].
  destruct
    (thread (fun news0 kh => apart_handler r news0 (snd kh))
       (skipn (binders_in_handler h) news) handlers)
    as [rest' handlers'].
  cbn [snd] in IHv, IHk |- *. cbn [flat_map snd]. rewrite !map_app.
  split; apply incl_app;
    first [ apply incl_appl; assumption | apply incl_appr; assumption ].
Qed.

Lemma flat_map_handlers {A : Type} (f : continuation_handler -> list A)
    (handlers : list (continuation * continuation_handler)) :
  flat_map (fun '(_, h) => f h) handlers
  = flat_map (fun kh => f (snd kh)) handlers.
Proof. apply flat_map_ext. intros [k h]. reflexivity. Qed.

Lemma let_cont_recursive_free
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun kh => apart_handler_free (snd kh)) handlers ->
  apart_free body ->
  apart_free (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers IHbody r news Hsupplied.
  pose proof Hsupplied as (Hnodup & Hcount & [Hov Hok]).
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hcount, Hov, Hok.
  unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  rewrite (flat_map_handlers free_variables_of_handler) in Hsupplied, Hov.
  rewrite (flat_map_handlers free_continuations_of_handler)
    in Hsupplied, Hok.
  assert (Hlength : length (map fst handlers) <= length news)
    by (rewrite length_map; lia).
  pose proof
    (supplied_bind_continuations (map fst handlers) r news _
       (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
        + binders body)
       _ _
       (flat_map (fun kh => free_continuations_of_handler (snd kh)) handlers
        ++ free_continuations_of_expr body)
       Hsupplied ltac:(rewrite length_map; lia)
       ltac:(intros j Hj; exact (in_without j (map fst handlers) _ Hj)))
    as H1.
  cbn [apart].
  rewrite (bind_continuations_eq (map fst handlers) r news Hlength).
  cbv beta iota.
  pose proof (bind_continuations_variable (map fst handlers) r news)
    as Hvariables.
  pose proof
    (bind_continuations_bound_in (map fst handlers) r news) as Hbound.
  pose proof (bind_continuations_unbound (map fst handlers) r news)
    as Hunbound.
  set (r1 := snd (bind_continuations r news (map fst handlers))) in *.
  set (rest0 := skipn (length (map fst handlers)) news) in *.
  destruct
    (thread_handlers_free r1 handlers IHhandlers rest0
       (flat_map (fun kh => free_variables_of_handler (snd kh)) handlers)
       (flat_map (fun kh => free_continuations_of_handler (snd kh)) handlers)
       (supplied_skipn 0 r1 rest0 _
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          _ _ _ _ H1 ltac:(lia) ltac:(apply incl_appl, incl_refl)
          ltac:(apply incl_appl, incl_refl)))
    as [Hhv Hhk].
  { intros kh Hin. split; intros a Ha; apply in_flat_map; exists kh;
      (split; [ exact Hin | exact Ha ]). }
  pose proof
    (thread_length (fun news0 kh => apart_handler r1 news0 (snd kh)) handlers
       rest0)
    as Hlen.
  pose proof (thread_handlers_rest r1 handlers rest0) as Hrest.
  destruct
    (thread (fun news0 kh => apart_handler r1 news0 (snd kh)) rest0 handlers)
    as [rest hs'].
  cbn [fst snd] in Hrest, Hhv, Hhk, Hlen. subst rest.
  destruct
    (IHbody r1 _
       (supplied_skipn
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          r1 rest0 _ _ _ _ _ _ H1 (le_n _)
          ltac:(apply incl_appr, incl_refl)
          ltac:(apply incl_appr, incl_refl)))
    as [IHv IHk].
  destruct
    (apart r1
       (skipn
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          rest0)
       body)
    as [rest' body'].
  cbn [snd] in IHv, IHk |- *.
  cbn [free_variables_of_expr free_continuations_of_expr].
  rewrite !(flat_map_handlers free_variables_of_handler),
    !(flat_map_handlers free_continuations_of_handler).
  assert (Hks :
    length (firstn (length (map fst handlers)) news) = length hs')
    by (rewrite Hlen, length_map; apply firstn_length_le;
        rewrite length_map in Hlength; exact Hlength).
  rewrite (flat_map_after free_variables_of_handler snd),
    (flat_map_after free_continuations_of_handler snd).
  rewrite combine_snd by exact Hks. rewrite combine_fst by exact Hks.
  split.
  - intros y Hy. rewrite map_app. apply in_or_app.
    apply in_app_or in Hy as [Hy | Hy];
      [ left; apply Hhv in Hy | right; apply IHv in Hy ];
      rewrite (map_ext _ _ Hvariables) in Hy; exact Hy.
  - intros j Hj. apply in_without_elim in Hj as [Hj Hne].
    assert (Hj' :
      In j
        (map (rename_continuation r1)
           (flat_map (fun kh => free_continuations_of_handler (snd kh))
              handlers
            ++ free_continuations_of_expr body))).
    { rewrite map_app. apply in_or_app.
      apply in_app_or in Hj as [Hj | Hj];
        [ left; exact (Hhk j Hj) | right; exact (IHk j Hj) ]. }
    apply in_map_iff in Hj' as (z & <- & Hz).
    destruct (in_dec Pos.eq_dec z (map fst handlers)) as [Hin | Hnotin].
    + exfalso. apply Hne. exact (Hbound z Hnodup Hlength Hin).
    + assert (Hfree :
        In z
          (without (map fst handlers)
             (flat_map (fun kh => free_continuations_of_handler (snd kh))
                handlers
              ++ free_continuations_of_expr body)))
        by (apply in_without_intro; [ exact Hz | exact Hnotin ]).
      rewrite (Hunbound z Hlength Hnotin); [ apply in_map; exact Hfree | ].
      intros Hin. exact (Hok z Hfree (in_firstn _ _ _ Hin)).
Qed.

Lemma code_free (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr) :
  apart_free body ->
  apart_code_free (Code params my_closure ret exn metadata body).
Proof.
  intros IH r news Hsupplied.
  pose proof Hsupplied as (Hnodup & Hcount & [Hov Hok]).
  cbn [binders_in_code free_variables_of_code free_continuations_of_code]
    in Hcount, Hov, Hok, Hsupplied.
  destruct news as [ | n news]; [ cbn [length] in Hcount; lia | ].
  cbn [length] in Hcount.
  assert (Hn : ~ In n news) by (inversion Hnodup; assumption).
  assert (Hnodup1 : NoDup news) by (inversion Hnodup; assumption).
  assert (Hlength : length params <= length news) by lia.
  pose proof
    (supplied_bind_variable my_closure r n news _
       (length params + S (S (binders body))) _
       (my_closure
        :: without (my_closure :: params) (free_variables_of_expr body))
       _ Hsupplied (le_n _)
       ltac:(intros y [Heq | Hin];
             [ left; symmetry; exact Heq | right; exact Hin ]))
    as H1.
  pose proof
    (supplied_bind_variables params _ news _ (S (S (binders body))) _
       (free_variables_of_expr body) _ H1 (le_n _)
       ltac:(intros y Hy;
             destruct (in_without y (my_closure :: params) _ Hy)
               as [[Heq | Hin] | Hin];
             [ right; left; exact Heq | left; exact Hin
             | right; right; exact Hin ]))
    as H2.
  cbn [apart_code new_variable hd tl].
  rewrite
    (bind_variables_eq params (bind_variable r my_closure n) news Hlength).
  cbv beta iota.
  pose proof
    (bind_variables_bound_in params (bind_variable r my_closure n) news)
    as Hbound.
  pose proof
    (bind_variables_unbound params (bind_variable r my_closure n) news)
    as Hunbound.
  pose proof
    (bind_variables_continuation params (bind_variable r my_closure n) news)
    as Hconts.
  set (rb := snd (bind_variables (bind_variable r my_closure n) news params))
    in *.
  assert (Hskip :
    incl (skipn (length params) news) news) by apply incl_skipn.
  destruct (skipn (length params) news) as [ | m1 [ | m2 rest]];
    [ destruct H2 as (_ & Hcount2 & _); cbn [length] in Hcount2; lia
    | destruct H2 as (_ & Hcount2 & _); cbn [length] in Hcount2; lia | ].
  cbn [new_continuation hd tl].
  pose proof
    (supplied_bind_continuation ret _ m1 (m2 :: rest) _ (S (binders body)) _
       _ (ret :: without [ret; exn] (free_continuations_of_expr body)) H2
       (le_n _)
       ltac:(intros j [Heq | Hin];
             [ left; symmetry; exact Heq | right; exact Hin ]))
    as H3.
  pose proof
    (supplied_bind_continuation exn _ m2 rest _ (binders body) _ _
       (free_continuations_of_expr body) H3 (le_n _)
       ltac:(intros j Hj;
             destruct (in_without j [ret; exn] _ Hj)
               as [[Heq | [Heq | []]] | Hin];
             [ right; left; exact Heq | left; symmetry; exact Heq
             | right; right; exact Hin ]))
    as H4.
  destruct (IH _ _ H4) as [IHv IHk].
  assert (Hm12 : m1 <> m2).
  { destruct H2 as (Hnodup2 & _ & _). inversion Hnodup2 as [ | ? ? Hnot _].
    intros Heq. apply Hnot. left. symmetry. exact Heq. }
  assert (Hm1 : In m1 news) by (apply Hskip; left; reflexivity).
  assert (Hm2 : In m2 news) by (apply Hskip; right; left; reflexivity).
  destruct
    (apart
       (bind_continuation (bind_continuation rb ret m1) exn m2) rest body)
    as [rest' body'].
  cbn [snd] in IHv, IHk |- *.
  cbn [free_variables_of_code free_continuations_of_code]. split.
  - intros y Hy. apply in_without_elim in Hy as [Hy Hne].
    apply IHv, in_map_iff in Hy as (z & <- & Hz).
    change
      (rename_variable
         (bind_continuation (bind_continuation rb ret m1) exn m2) z)
      with (rename_variable rb z) in Hne |- *.
    destruct (in_dec Pos.eq_dec z params) as [Hin | Hnotin].
    + exfalso. apply Hne. right.
      exact (Hbound z Hnodup1 Hlength Hin).
    + destruct (Pos.eq_dec z my_closure) as [-> | Hzc].
      * exfalso. apply Hne. left.
        rewrite (Hunbound my_closure Hlength Hnotin).
        -- rewrite rename_variable_bind_variable, swap_first. reflexivity.
        -- rewrite rename_variable_bind_variable, swap_first. intros Hin.
           exact (Hn (in_firstn _ _ _ Hin)).
      * assert (Hfree :
          In z
            (without (my_closure :: params) (free_variables_of_expr body))).
        { apply in_without_intro; [ exact Hz | ].
          intros [Heq | Hin];
            [ apply Hzc; symmetry; exact Heq | exact (Hnotin Hin) ]. }
        assert (Hother :
          rename_variable (bind_variable r my_closure n) z
          = rename_variable r z).
        { apply bind_variable_other; [ exact Hzc | ].
          intros Heq. apply (Hov z Hfree). left. symmetry. exact Heq. }
        rewrite (Hunbound z Hlength Hnotin), Hother;
          [ apply in_map; exact Hfree | ].
        rewrite Hother. intros Hin.
        apply (Hov z Hfree). right. exact (in_firstn _ _ _ Hin).
  - intros j Hj. apply in_without_elim in Hj as [Hj Hne].
    apply IHk, in_map_iff in Hj as (z & <- & Hz).
    destruct (Pos.eq_dec z exn) as [-> | Hze].
    + exfalso. apply Hne. right. left.
      rewrite rename_continuation_bind_continuation, swap_first.
      reflexivity.
    + destruct (Pos.eq_dec z ret) as [-> | Hzr].
      * exfalso. apply Hne. left.
        rewrite (bind_continuation_other _ exn ret m2 Hze);
          rewrite rename_continuation_bind_continuation, swap_first;
          [ reflexivity | exact Hm12 ].
      * assert (Hfree :
          In z (without [ret; exn] (free_continuations_of_expr body))).
        { apply in_without_intro; [ exact Hz | ].
          intros [Heq | [Heq | []]];
            [ apply Hzr | apply Hze ]; symmetry; exact Heq. }
        assert (Hb : rename_continuation rb z = rename_continuation r z)
          by exact (Hconts z).
        assert (Hret :
          rename_continuation (bind_continuation rb ret m1) z
          = rename_continuation r z).
        { rewrite (bind_continuation_other rb ret z m1 Hzr); [ exact Hb | ].
          rewrite Hb. intros Heq. apply (Hok z Hfree). right.
          rewrite Heq. exact Hm1. }
        rewrite (bind_continuation_other _ exn z m2 Hze), Hret;
          [ apply in_map; exact Hfree | ].
        rewrite Hret. intros Heq. apply (Hok z Hfree). right.
        rewrite Heq. exact Hm2.
Qed.

Lemma bind_code_free (id : code_id) (c : code) :
  apart_code_free c -> apart_binding_free (Bind_code id c).
Proof.
  intros IH r news Hsupplied. cbn [apart_binding].
  specialize (IH r news Hsupplied).
  destruct (apart_code r news c) as [rest c']. exact IH.
Qed.

Lemma bind_symbol_free (s : symbol) (const : static_const) :
  apart_binding_free (Bind_symbol s const).
Proof.
  intros r news _. cbn [apart_binding snd].
  cbn [free_variables_of_binding free_continuations_of_binding].
  rewrite free_variables_of_static_const_renamed.
  split; [ apply incl_refl | intros a [] ].
Qed.

Lemma apart_free_all :
  (∀ e : expr, apart_free e)
  ∧ (∀ h : continuation_handler, apart_handler_free h)
  ∧ (∀ binding : static_binding, apart_binding_free binding)
  ∧ (∀ c : code, apart_code_free c).
Proof.
  apply term_mutual_ind.
  - exact let_free.
  - exact let_static_free.
  - exact let_cont_free.
  - exact let_cont_recursive_free.
  - intros ap r news _. apply renamed_free. reflexivity.
  - intros ac r news _. apply renamed_free. reflexivity.
  - intros scrutinee arms r news _. apply renamed_free. reflexivity.
  - intros r news _. apply renamed_free. reflexivity.
  - exact handler_free.
  - exact bind_code_free.
  - intros id r news _. split; intros a [].
  - exact bind_symbol_free.
  - exact code_free.
Qed.

Lemma renamed_apart_supplied (avoid : list continuation) (e : expr) :
  supplied identity_renaming
    (fresh_names (binders e) (avoid ++ names_of e)) e.
Proof.
  destruct (fresh_names_fresh (binders e) (avoid ++ names_of e))
    as [Hlength [Hnodup Houtside]].
  split; [ apply distinct_NoDup, Hnodup | ].
  split; [ rewrite Hlength; apply le_n | ].
  unfold names_of in Houtside. split.
  - intros x Hx Hin.
    pose proof (Houtside x (proj2 (among_In _ _) Hin)) as Hnot.
    rewrite !among_app,
      (proj2 (among_In x _) (free_variables_in_variables_of e x Hx)),
      orb_true_r
      in Hnot.
    discriminate Hnot.
  - intros k Hk Hin.
    pose proof (Houtside k (proj2 (among_In _ _) Hin)) as Hnot.
    rewrite !among_app,
      (proj2 (among_In k _) (free_continuations_in_continuations_of e k Hk)),
      !orb_true_r
      in Hnot.
    discriminate Hnot.
Qed.

Theorem renamed_apart_avoiding_free_names :
  renamed_apart_avoiding_free_names_statement.
Proof.
  intros avoid e.
  destruct
    (proj1 apart_free_all e identity_renaming _
       (renamed_apart_supplied avoid e))
    as [Hv Hk].
  split; intros a Ha; [ apply Hv in Ha | apply Hk in Ha ];
    apply in_map_iff in Ha as (z & <- & Hz); exact Hz.
Qed.

Theorem canonical_avoiding_free_names :
  canonical_avoiding_free_names_statement.
Proof.
  intros avoid e.
  destruct
    (proj1 apart_free_all e identity_renaming _ (canonical_supplied avoid e))
    as [Hv Hk].
  split; intros a Ha; [ apply Hv in Ha | apply Hk in Ha ];
    apply in_map_iff in Ha as (z & <- & Hz); exact Hz.
Qed.

(* ======================================================================== *)
(* The canonical form of a renamed term                                     *)
(* ======================================================================== *)

(* The renaming is the renaming of the term and then swaps that fix what
   it makes of the free names of the term and of the continuations to
   avoid, and send what it makes of each new name of the term to the
   new name at the same place of the renamed term. Both canonical forms
   are what [apart] gives of the term under that renaming with the new
   names of the renamed term. *)
Theorem canonical_avoiding_equivariant_up_to :
  canonical_avoiding_equivariant_up_to_statement.
Proof.
  intros r avoid e.
  destruct (canonical_avoiding_free_names avoid e) as [Hfree_v Hfree_k].
  pose proof
    (fresh_names_length (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    as Hlength.
  destruct
    (fresh_names_outside (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    as [Hdistinct Houtside].
  pose proof
    (fresh_names_length (binders (rename_expr r e))
       (map (rename_continuation r) avoid
        ++ free_variables_of_expr (rename_expr r e)
        ++ free_continuations_of_expr (rename_expr r e)))
    as Hlength'.
  destruct
    (fresh_names_outside (binders (rename_expr r e))
       (map (rename_continuation r) avoid
        ++ free_variables_of_expr (rename_expr r e)
        ++ free_continuations_of_expr (rename_expr r e)))
    as [Hdistinct' Houtside'].
  assert (Hcandidate :
    canonical_avoiding avoid e
    = snd
        (apart identity_renaming
           (fresh_names (binders e)
              (avoid ++ free_variables_of_expr e
               ++ free_continuations_of_expr e))
           e))
    by reflexivity.
  assert (Hcandidate' :
    canonical_avoiding (map (rename_continuation r) avoid) (rename_expr r e)
    = snd
        (apart identity_renaming
           (fresh_names (binders (rename_expr r e))
              (map (rename_continuation r) avoid
               ++ free_variables_of_expr (rename_expr r e)
               ++ free_continuations_of_expr (rename_expr r e)))
           (rename_expr r e)))
    by reflexivity.
  revert Hlength Hdistinct Houtside Hlength' Hdistinct' Houtside' Hcandidate
    Hcandidate'.
  generalize
    (fresh_names (binders e)
       (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
    as news.
  generalize
    (fresh_names (binders (rename_expr r e))
       (map (rename_continuation r) avoid
        ++ free_variables_of_expr (rename_expr r e)
        ++ free_continuations_of_expr (rename_expr r e)))
    as news'.
  intros news' news Hlength Hdistinct Houtside Hlength' Hdistinct' Houtside'
    Hcandidate Hcandidate'.
  rewrite binders_renamed in Hlength'.
  rewrite free_variables_of_expr_renamed, free_continuations_of_expr_renamed
    in Houtside'.
  (* The variables. *)
  destruct
    (exchange (map (rename_variable r) (free_variables_of_expr e))
       (map (rename_variable r) news) news')
    as (swaps_v & Hfixes_v & Hmoves_v).
  { rewrite length_map.
    transitivity (binders e); [ exact Hlength | symmetry; exact Hlength' ]. }
  { rewrite (distinct_map_injective _ (rename_variable_eqb r)).
    exact Hdistinct. }
  { exact Hdistinct'. }
  { intros name Hname. apply among_In, in_map_iff in Hname.
    destruct Hname as (a & <- & Ha).
    rewrite (among_map_injective _ (rename_variable_eqb r)).
    apply among_In, Houtside, among_app_false in Ha.
    destruct Ha as [_ Ha]. apply among_app_false in Ha. exact (proj1 Ha). }
  { intros name Hname. apply Houtside', among_app_false in Hname.
    destruct Hname as [_ Hname]. apply among_app_false in Hname.
    exact (proj1 Hname). }
  (* The continuations, those to avoid among them. *)
  destruct
    (exchange
       (map (rename_continuation r)
          (avoid ++ free_continuations_of_expr e))
       (map (rename_continuation r) news) news')
    as (swaps_k & Hfixes_k & Hmoves_k).
  { rewrite length_map.
    transitivity (binders e); [ exact Hlength | symmetry; exact Hlength' ]. }
  { rewrite (distinct_map_injective _ (rename_continuation_eqb r)).
    exact Hdistinct. }
  { exact Hdistinct'. }
  { intros name Hname. apply among_In, in_map_iff in Hname.
    destruct Hname as (a & <- & Ha).
    rewrite (among_map_injective _ (rename_continuation_eqb r)).
    apply among_In, Houtside, among_app_false in Ha.
    destruct Ha as [Havoid Ha]. apply among_app_false in Ha.
    apply among_app_false. split; [ exact Havoid | exact (proj2 Ha) ]. }
  { intros name Hname. apply Houtside', among_app_false in Hname.
    destruct Hname as [Havoid Hname]. apply among_app_false in Hname.
    rewrite map_app. apply among_app_false.
    split; [ exact Havoid | exact (proj2 Hname) ]. }
  (* The renaming: [r], then the swaps. *)
  set (s :=
    Renaming (swaps_v ++ variable_permutation r)
      (swaps_k ++ continuation_permutation r)).
  assert (Hs_v :
    ∀ x : variable,
      rename_variable s x = permute swaps_v (rename_variable r x))
    by (intros x; apply permute_app).
  assert (Hs_k :
    ∀ k : continuation,
      rename_continuation s k = permute swaps_k (rename_continuation r k))
    by (intros k; apply permute_app).
  assert (Hsame_v :
    ∀ x : variable,
      In x (free_variables_of_expr e) ->
      rename_variable r x = rename_variable s x).
  { intros x Hx. rewrite Hs_v. symmetry. apply Hfixes_v, among_In, in_map, Hx.
  }
  assert (Hsame_fixed :
    ∀ k : continuation,
      In k (avoid ++ free_continuations_of_expr e) ->
      rename_continuation r k = rename_continuation s k).
  { intros k Hk. rewrite Hs_k. symmetry.
    apply Hfixes_k, among_In, in_map, Hk. }
  assert (Hsame_k :
    ∀ k : continuation,
      In k (free_continuations_of_expr e) ->
      rename_continuation r k = rename_continuation s k)
    by (intros k Hk; apply Hsame_fixed, in_or_app; right; exact Hk).
  assert (Hnews_v : map (rename_variable s) news = news').
  { rewrite (map_ext _ _ Hs_v),
      <- (map_map (rename_variable r) (permute swaps_v)).
    exact Hmoves_v. }
  assert (Hnews_k : map (rename_continuation s) news = news').
  { rewrite (map_ext _ _ Hs_k),
      <- (map_map (rename_continuation r) (permute swaps_k)).
    exact Hmoves_k. }
  assert (Hagree : agree s news).
  { intros a Ha.
    exact
      (map_same (rename_variable s) (rename_continuation s) news
         (eq_trans Hnews_v (eq_sym Hnews_k)) a Ha). }
  (* The new names of the renamed term are enough for the term under the
     renaming, and outside what it makes of its free names. *)
  assert (Hnodup : NoDup news') by (apply distinct_NoDup, Hdistinct').
  assert (Hcount : binders e <= length news')
    by (rewrite Hlength'; apply le_n).
  assert (Hout :
    outside s news' (free_variables_of_expr e)
      (free_continuations_of_expr e)).
  { split.
    - intros x Hx Hin. rewrite <- (Hsame_v x Hx) in Hin.
      pose proof (Houtside' _ (proj2 (among_In _ _) Hin)) as Hnot.
      rewrite !among_app,
        (proj2 (among_In _ _)
           (in_map (rename_variable r) _ x Hx)),
        orb_true_r
        in Hnot.
      discriminate Hnot.
    - intros k Hk Hin. rewrite <- (Hsame_k k Hk) in Hin.
      pose proof (Houtside' _ (proj2 (among_In _ _) Hin)) as Hnot.
      rewrite !among_app,
        (proj2 (among_In _ _)
           (in_map (rename_continuation r) _ k Hk)),
        !orb_true_r
        in Hnot.
      discriminate Hnot. }
  assert (Hleft :
    apart identity_renaming news' (rename_expr r e) = apart s news' e).
  { apply (proj1 apart_fused_all e r identity_renaming s news' Hnodup Hcount);
      [ | exact Hout ].
    split; [ intros x Hx; symmetry; exact (Hsame_v x Hx) | ].
    intros k Hk. symmetry. exact (Hsame_k k Hk). }
  assert (Hright :
    apart identity_renaming news' (rename_expr s e) = apart s news' e).
  { apply (proj1 apart_fused_all e s identity_renaming s news' Hnodup Hcount);
      [ | exact Hout ].
    split; intros a _; reflexivity. }
  assert (Hrenamed :
    snd (apart identity_renaming news' (rename_expr s e))
    = rename_expr s (snd (apart identity_renaming news e))).
  { rewrite <- Hnews_v.
    rewrite
      (apart_renamed s e identity_renaming identity_renaming news
         (carries_identity s) Hagree).
    reflexivity. }
  exists s. split; [ | split; [ | split ] ].
  - intros x Hx. symmetry. apply Hsame_v, Hfree_v, among_In, Hx.
  - intros k Hk. symmetry. apply Hsame_k, Hfree_k, among_In, Hk.
  - intros k Hk. symmetry. apply Hsame_fixed, in_or_app. left.
    apply among_In, Hk.
  - rewrite Hcandidate', Hcandidate, <- Hrenamed, Hleft, Hright.
    reflexivity.
Qed.

(* ======================================================================== *)
(* The candidate of a term with closed code has closed code                 *)
(* ======================================================================== *)

Definition apart_closed (e : expr) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied r news e ->
    code_closed e = true ->
    code_closed (snd (apart r news e)) = true.

Definition apart_handler_closed (h : continuation_handler) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied_names r news (binders_in_handler h)
      (free_variables_of_handler h) (free_continuations_of_handler h) ->
    handler_code_closed h = true ->
    handler_code_closed (snd (apart_handler r news h)) = true.

Definition apart_binding_closed (binding : static_binding) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied_names r news (binders_in_binding binding)
      (free_variables_of_binding binding)
      (free_continuations_of_binding binding) ->
    binding_code_closed binding = true ->
    binding_code_closed (snd (apart_binding r news binding)) = true.

Definition apart_code_closed (c : code) : Prop :=
  ∀ (r : renaming) (news : list positive),
    supplied_names r news (binders_in_code c) (free_variables_of_code c)
      (free_continuations_of_code c) ->
    code_is_closed c = true ->
    code_is_closed (snd (apart_code r news c)) = true.

Lemma let_closed (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  apart_closed body -> apart_closed (Let x kind d body).
Proof.
  intros IH r news Hsupplied Hclosed. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  destruct news as [ | n news];
    [ destruct Hsupplied as (_ & Hcount & _); cbn [length] in Hcount; lia
    | ].
  assert (Hbody : supplied (bind_variable r x n) news body).
  { apply (supplied_bind_variable x r n news _ _ _ _ _ Hsupplied (le_n _)).
    intros y Hy.
    destruct (in_without_one y x _ Hy) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ]. }
  specialize (IH _ _ Hbody Hclosed). cbn [apart new_variable hd tl].
  destruct (apart (bind_variable r x n) news body) as [rest body'].
  exact IH.
Qed.

Lemma handler_closed (params : list variable) (metadata : handler_metadata)
    (body : expr) :
  apart_closed body -> apart_handler_closed (Handler params metadata body).
Proof.
  intros IH r news Hsupplied Hclosed.
  cbn [binders_in_handler free_variables_of_handler
       free_continuations_of_handler]
    in Hsupplied.
  assert (Hlength : length params <= length news)
    by (destruct Hsupplied as (_ & Hcount & _); lia).
  assert (Hbody :
    supplied (snd (bind_variables r news params))
      (skipn (length params) news) body).
  { apply
      (supplied_bind_variables params r news _ _ _ _ _ Hsupplied (le_n _)).
    intros y Hy. exact (in_without y params _ Hy). }
  specialize (IH _ _ Hbody Hclosed). cbn [apart_handler].
  rewrite (bind_variables_eq params r news Hlength). cbv beta iota.
  destruct
    (apart (snd (bind_variables r news params)) (skipn (length params) news)
       body)
    as [rest body'].
  exact IH.
Qed.

Lemma let_cont_closed (k : continuation) (h : continuation_handler)
    (body : expr) :
  apart_handler_closed h -> apart_closed body ->
  apart_closed (Let_cont_non_recursive k h body).
Proof.
  intros IHh IHbody r news Hsupplied Hclosed. unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  cbn [code_closed] in Hclosed.
  apply andb_true_iff in Hclosed as [Hclosed_h Hclosed_body].
  specialize
    (IHh r news
       (supplied_skipn 0 r news _ (binders_in_handler h) _
          (free_variables_of_handler h) _ (free_continuations_of_handler h)
          Hsupplied ltac:(lia) ltac:(apply incl_appl, incl_refl)
          ltac:(apply incl_appl, incl_refl))
       Hclosed_h).
  cbn [apart].
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, IHh. subst rest.
  pose proof
    (supplied_skipn (binders_in_handler h) r news _
       (S (binders body)) _ (free_variables_of_expr body) _
       (free_continuations_of_handler h
        ++ without [k] (free_continuations_of_expr body))
       Hsupplied (le_n _) ltac:(apply incl_appr, incl_refl)
       ltac:(apply incl_refl))
    as H1.
  destruct (skipn (binders_in_handler h) news) as [ | m rest];
    [ destruct H1 as (_ & Hcount & _); cbn [length] in Hcount; lia | ].
  cbn [new_continuation hd tl].
  assert (Hbody : supplied (bind_continuation r k m) rest body).
  { apply (supplied_bind_continuation k r m rest _ _ _ _ _ H1 (le_n _)).
    intros j Hj.
    destruct (in_without_one j k _ Hj) as [Heq | Hin];
      [ left; exact Heq | right; apply in_or_app; right; exact Hin ]. }
  specialize (IHbody _ _ Hbody Hclosed_body).
  destruct (apart (bind_continuation r k m) rest body) as [rest' body'].
  cbn [snd] in IHbody |- *. cbn [code_closed]. rewrite IHh, IHbody.
  reflexivity.
Qed.

Lemma thread_bindings_closed (r : renaming)
    (bindings : list static_binding) :
  Forall apart_binding_closed bindings ->
  ∀ (news : list positive) (vs : list variable) (ks : list continuation),
    supplied_names r news (list_sum (map binders_in_binding bindings)) vs ks
    ->
    (∀ binding : static_binding,
      In binding bindings ->
      incl (free_variables_of_binding binding) vs
      ∧ incl (free_continuations_of_binding binding) ks) ->
    forallb binding_code_closed bindings = true ->
    forallb binding_code_closed
      (snd (thread (apart_binding r) news bindings))
    = true.
Proof.
  intros Hall.
  induction Hall as [ | binding bindings Hb _ IH];
    intros news vs ks Hsupplied Hincl Hclosed;
    [ reflexivity | ].
  cbn [map] in Hsupplied. rewrite list_sum_cons in Hsupplied.
  cbn [forallb] in Hclosed.
  apply andb_true_iff in Hclosed as [Hclosed_b Hclosed_bs].
  destruct (Hincl binding (or_introl eq_refl)) as [Hvs Hks].
  specialize
    (Hb r news
       (supplied_skipn 0 r news _ (binders_in_binding binding) _ _ _ _
          Hsupplied ltac:(lia) Hvs Hks)
       Hclosed_b).
  cbn [thread].
  pose proof (apart_binding_rest binding r news) as Hrest.
  destruct (apart_binding r news binding) as [rest binding'].
  cbn [fst snd] in Hrest, Hb. subst rest.
  specialize
    (IH (skipn (binders_in_binding binding) news) vs ks
       (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _)
          (incl_refl _) (incl_refl _))
       (fun b Hin => Hincl b (or_intror Hin)) Hclosed_bs).
  destruct
    (thread (apart_binding r) (skipn (binders_in_binding binding) news)
       bindings)
    as [rest' bindings'].
  cbn [snd] in IH |- *. cbn [forallb]. rewrite Hb, IH. reflexivity.
Qed.

Lemma let_static_closed (bindings : list static_binding) (body : expr) :
  Forall apart_binding_closed bindings -> apart_closed body ->
  apart_closed (Let_static bindings body).
Proof.
  intros IHbindings IHbody r news Hsupplied Hclosed.
  unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  cbn [code_closed] in Hclosed.
  apply andb_true_iff in Hclosed as [Hclosed_bs Hclosed_body].
  pose proof
    (thread_bindings_closed r bindings IHbindings news
       (flat_map free_variables_of_binding bindings
        ++ free_variables_of_expr body)
       (flat_map free_continuations_of_binding bindings
        ++ free_continuations_of_expr body)
       (supplied_skipn 0 r news _ (list_sum (map binders_in_binding bindings))
          _ _ _ _ Hsupplied ltac:(lia) (incl_refl _) (incl_refl _)))
    as Hbs.
  assert (Hincl :
    ∀ binding : static_binding,
      In binding bindings ->
      incl (free_variables_of_binding binding)
        (flat_map free_variables_of_binding bindings
         ++ free_variables_of_expr body)
      ∧ incl (free_continuations_of_binding binding)
          (flat_map free_continuations_of_binding bindings
           ++ free_continuations_of_expr body)).
  { intros binding Hin. split; intros a Ha; apply in_or_app; left;
      apply in_flat_map; exists binding; (split; [ exact Hin | exact Ha ]). }
  specialize (Hbs Hincl Hclosed_bs).
  cbn [apart].
  pose proof (thread_bindings_rest r bindings news) as Hrest.
  destruct (thread (apart_binding r) news bindings) as [rest bindings'].
  cbn [fst snd] in Hrest, Hbs. subst rest.
  specialize
    (IHbody r (skipn (list_sum (map binders_in_binding bindings)) news)
       (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _)
          ltac:(apply incl_appr, incl_refl)
          ltac:(apply incl_appr, incl_refl))
       Hclosed_body).
  destruct
    (apart r (skipn (list_sum (map binders_in_binding bindings)) news) body)
    as [rest' body'].
  cbn [snd] in IHbody |- *. cbn [code_closed]. rewrite Hbs, IHbody.
  reflexivity.
Qed.

Lemma thread_handlers_closed (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun kh => apart_handler_closed (snd kh)) handlers ->
  ∀ (news : list positive) (vs : list variable) (ks : list continuation),
    supplied_names r news
      (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)) vs
      ks ->
    (∀ kh : continuation * continuation_handler,
      In kh handlers ->
      incl (free_variables_of_handler (snd kh)) vs
      ∧ incl (free_continuations_of_handler (snd kh)) ks) ->
    forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
    forallb handler_code_closed
      (snd
         (thread (fun news kh => apart_handler r news (snd kh)) news
            handlers))
    = true.
Proof.
  intros Hall.
  induction Hall as [ | [k h] handlers Hh _ IH];
    intros news vs ks Hsupplied Hincl Hclosed;
    [ reflexivity | ].
  cbn [map snd] in Hsupplied, Hh. rewrite list_sum_cons in Hsupplied.
  cbn [forallb] in Hclosed.
  apply andb_true_iff in Hclosed as [Hclosed_h Hclosed_hs].
  destruct (Hincl (k, h) (or_introl eq_refl)) as [Hvs Hks].
  cbn [snd] in Hvs, Hks.
  specialize
    (Hh r news
       (supplied_skipn 0 r news _ (binders_in_handler h) _ _ _ _ Hsupplied
          ltac:(lia) Hvs Hks)
       Hclosed_h).
  cbn [thread snd].
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, Hh. subst rest.
  specialize
    (IH (skipn (binders_in_handler h) news) vs ks
       (supplied_skipn _ r news _ _ _ _ _ _ Hsupplied (le_n _)
          (incl_refl _) (incl_refl _))
       (fun kh Hin => Hincl kh (or_intror Hin)) Hclosed_hs).
  destruct
    (thread (fun news0 kh => apart_handler r news0 (snd kh))
       (skipn (binders_in_handler h) news) handlers)
    as [rest' handlers'].
  cbn [snd] in IH |- *. cbn [forallb]. rewrite Hh, IH. reflexivity.
Qed.

Lemma forallb_handlers_combine {A : Type}
    (f : continuation_handler -> bool) (ks : list A) :
  ∀ hs : list continuation_handler,
    length ks = length hs ->
    forallb (fun '(_, h) => f h) (combine ks hs) = forallb f hs.
Proof.
  induction ks as [ | k ks IH]; intros [ | h hs] Hlength;
    try reflexivity; try discriminate Hlength.
  cbn [combine forallb]. rewrite IH; [ reflexivity | ].
  injection Hlength as Hlength. exact Hlength.
Qed.

Lemma let_cont_recursive_closed
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun kh => apart_handler_closed (snd kh)) handlers ->
  apart_closed body ->
  apart_closed (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers IHbody r news Hsupplied Hclosed.
  unfold supplied in Hsupplied.
  cbn [binders free_variables_of_expr free_continuations_of_expr]
    in Hsupplied.
  rewrite (flat_map_handlers free_variables_of_handler) in Hsupplied.
  rewrite (flat_map_handlers free_continuations_of_handler) in Hsupplied.
  cbn [code_closed] in Hclosed.
  apply andb_true_iff in Hclosed as [Hclosed_hs Hclosed_body].
  assert (Hlength : length (map fst handlers) <= length news)
    by (destruct Hsupplied as (_ & Hcount & _); rewrite length_map; lia).
  pose proof
    (supplied_bind_continuations (map fst handlers) r news _
       (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
        + binders body)
       _ _
       (flat_map (fun kh => free_continuations_of_handler (snd kh)) handlers
        ++ free_continuations_of_expr body)
       Hsupplied ltac:(rewrite length_map; lia)
       ltac:(intros j Hj; exact (in_without j (map fst handlers) _ Hj)))
    as H1.
  cbn [apart].
  rewrite (bind_continuations_eq (map fst handlers) r news Hlength).
  cbv beta iota.
  set (r1 := snd (bind_continuations r news (map fst handlers))) in *.
  set (rest0 := skipn (length (map fst handlers)) news) in *.
  pose proof
    (thread_handlers_closed r1 handlers IHhandlers rest0
       (flat_map (fun kh => free_variables_of_handler (snd kh)) handlers)
       (flat_map (fun kh => free_continuations_of_handler (snd kh)) handlers)
       (supplied_skipn 0 r1 rest0 _
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          _ _ _ _ H1 ltac:(lia) ltac:(apply incl_appl, incl_refl)
          ltac:(apply incl_appl, incl_refl)))
    as Hhs.
  assert (Hincl :
    ∀ kh : continuation * continuation_handler,
      In kh handlers ->
      incl (free_variables_of_handler (snd kh))
        (flat_map (fun kh => free_variables_of_handler (snd kh)) handlers)
      ∧ incl (free_continuations_of_handler (snd kh))
          (flat_map (fun kh => free_continuations_of_handler (snd kh))
             handlers)).
  { intros kh Hin. split; intros a Ha; apply in_flat_map; exists kh;
      (split; [ exact Hin | exact Ha ]). }
  specialize (Hhs Hincl Hclosed_hs).
  pose proof
    (thread_length (fun news0 kh => apart_handler r1 news0 (snd kh)) handlers
       rest0)
    as Hlen.
  pose proof (thread_handlers_rest r1 handlers rest0) as Hrest.
  destruct
    (thread (fun news0 kh => apart_handler r1 news0 (snd kh)) rest0 handlers)
    as [rest hs'].
  cbn [fst snd] in Hrest, Hhs, Hlen. subst rest.
  specialize
    (IHbody r1 _
       (supplied_skipn
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          r1 rest0 _ _ _ _ _ _ H1 (le_n _)
          ltac:(apply incl_appr, incl_refl)
          ltac:(apply incl_appr, incl_refl))
       Hclosed_body).
  destruct
    (apart r1
       (skipn
          (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
          rest0)
       body)
    as [rest' body'].
  cbn [snd] in IHbody |- *. cbn [code_closed].
  rewrite forallb_handlers_combine; [ rewrite Hhs, IHbody; reflexivity | ].
  rewrite Hlen, length_map. apply firstn_length_le.
  rewrite length_map in Hlength. exact Hlength.
Qed.

Lemma code_is_closed_parts (c : code) :
  free_variables_of_code c = [] ->
  free_continuations_of_code c = [] ->
  code_closed (code_body c) = true ->
  code_is_closed c = true.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [code_is_closed code_body]. intros -> -> ->. reflexivity.
Qed.

Lemma code_closed_apart (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr) :
  apart_closed body ->
  apart_code_closed (Code params my_closure ret exn metadata body).
Proof.
  intros IH r news Hsupplied Hclosed.
  destruct
    (proj2 (proj2 (proj2 apart_free_all))
       (Code params my_closure ret exn metadata body) r news Hsupplied)
    as [Hfree_v Hfree_k].
  destruct (code_closed_no_free_name _ Hclosed) as [Hv Hk].
  rewrite Hv in Hfree_v. rewrite Hk in Hfree_k.
  cbn [code_is_closed] in Hclosed.
  apply andb_true_iff in Hclosed as [_ Hclosed_body].
  apply code_is_closed_parts.
  - destruct
      (free_variables_of_code
         (snd
            (apart_code r news
               (Code params my_closure ret exn metadata body))))
      as [ | a l];
      [ reflexivity | destruct (Hfree_v a (or_introl eq_refl)) ].
  - destruct
      (free_continuations_of_code
         (snd
            (apart_code r news
               (Code params my_closure ret exn metadata body))))
      as [ | a l];
      [ reflexivity | destruct (Hfree_k a (or_introl eq_refl)) ].
  - clear Hfree_v Hfree_k Hv Hk.
    cbn [binders_in_code free_variables_of_code free_continuations_of_code]
      in Hsupplied.
    destruct news as [ | n news];
      [ destruct Hsupplied as (_ & Hcount & _); cbn [length] in Hcount; lia
      | ].
    pose proof
      (supplied_bind_variable my_closure r n news _
         (length params + S (S (binders body))) _
         (my_closure
          :: without (my_closure :: params) (free_variables_of_expr body))
         _ Hsupplied (le_n _)
         ltac:(intros y [Heq | Hin];
               [ left; symmetry; exact Heq | right; exact Hin ]))
      as H1.
    assert (Hlength : length params <= length news)
      by (destruct H1 as (_ & Hcount & _); lia).
    pose proof
      (supplied_bind_variables params _ news _ (S (S (binders body))) _
         (free_variables_of_expr body) _ H1 (le_n _)
         ltac:(intros y Hy;
               destruct (in_without y (my_closure :: params) _ Hy)
                 as [[Heq | Hin] | Hin];
               [ right; left; exact Heq | left; exact Hin
               | right; right; exact Hin ]))
      as H2.
    cbn [apart_code new_variable hd tl].
    rewrite
      (bind_variables_eq params (bind_variable r my_closure n) news Hlength).
    cbv beta iota.
    destruct (skipn (length params) news) as [ | m1 [ | m2 rest]];
      [ destruct H2 as (_ & Hcount & _); cbn [length] in Hcount; lia
      | destruct H2 as (_ & Hcount & _); cbn [length] in Hcount; lia | ].
    cbn [new_continuation hd tl].
    pose proof
      (supplied_bind_continuation ret _ m1 (m2 :: rest) _ (S (binders body))
         _ _ (ret :: without [ret; exn] (free_continuations_of_expr body))
         H2 (le_n _)
         ltac:(intros j [Heq | Hin];
               [ left; symmetry; exact Heq | right; exact Hin ]))
      as H3.
    pose proof
      (supplied_bind_continuation exn _ m2 rest _ (binders body) _ _
         (free_continuations_of_expr body) H3 (le_n _)
         ltac:(intros j Hj;
               destruct (in_without j [ret; exn] _ Hj)
                 as [[Heq | [Heq | []]] | Hin];
               [ right; left; exact Heq | left; symmetry; exact Heq
               | right; right; exact Hin ]))
      as H4.
    specialize (IH _ _ H4 Hclosed_body).
    match goal with
    | |- context [apart ?r4 rest body] =>
      destruct (apart r4 rest body) as [rest' body']
    end.
    exact IH.
Qed.

Lemma bind_code_closed (id : code_id) (c : code) :
  apart_code_closed c -> apart_binding_closed (Bind_code id c).
Proof.
  intros IH r news Hsupplied Hclosed. cbn [apart_binding].
  specialize (IH r news Hsupplied Hclosed).
  destruct (apart_code r news c) as [rest c']. exact IH.
Qed.

Lemma apart_closed_all :
  (∀ e : expr, apart_closed e)
  ∧ (∀ h : continuation_handler, apart_handler_closed h)
  ∧ (∀ binding : static_binding, apart_binding_closed binding)
  ∧ (∀ c : code, apart_code_closed c).
Proof.
  apply term_mutual_ind.
  - exact let_closed.
  - exact let_static_closed.
  - exact let_cont_closed.
  - exact let_cont_recursive_closed.
  - intros ap r news _ _. reflexivity.
  - intros ac r news _ _. reflexivity.
  - intros scrutinee arms r news _ _. reflexivity.
  - intros r news _ _. reflexivity.
  - exact handler_closed.
  - exact bind_code_closed.
  - intros id r news _ _. reflexivity.
  - intros s const r news _ _. reflexivity.
  - exact code_closed_apart.
Qed.

Theorem renamed_apart_avoiding_code_closed :
  renamed_apart_avoiding_code_closed_statement.
Proof.
  intros avoid e.
  exact
    (proj1 apart_closed_all e identity_renaming _
       (renamed_apart_supplied avoid e)).
Qed.

Theorem canonical_avoiding_code_closed :
  canonical_avoiding_code_closed_statement.
Proof.
  intros avoid e.
  exact
    (proj1 apart_closed_all e identity_renaming _
       (canonical_supplied avoid e)).
Qed.

(* ======================================================================== *)
(* No second renaming does to the bound names what the first does           *)
(* ======================================================================== *)

(* [let x = 0 in cont k (x)], with [k] the name 1 and [x] the name 2:
   [x] is the first name above the free names of the term, so the term
   is its own canonical form. *)
Definition own_canonical_form : expr :=
  Let 2%positive Value (Simple (Const (Tagged_immediate 0%Z)))
    (Apply_cont (Apply_cont_expr 1%positive [Var 2%positive] None)).

(* The swap of the variables 2 and 3. *)
Definition swap_of_bound_name : renaming :=
  Renaming [(2%positive, 3%positive)] [].

(* The term with [x] renamed has the free names of the term, so its
   canonical form is the term too; a second renaming that does to [x]
   what the swap does would give a term that binds 3. *)
Theorem canonical_avoiding_not_equivariant_up_to_names :
  canonical_avoiding_not_equivariant_up_to_names_statement.
Proof.
  intros H.
  destruct (H swap_of_bound_name own_canonical_form) as [r' [Hv [_ Heq]]].
  assert
    (Hsame : canonical_avoiding [] own_canonical_form = own_canonical_form)
    by (vm_compute; reflexivity).
  assert
    (Hrenamed :
      canonical_avoiding []
        (rename_expr swap_of_bound_name own_canonical_form)
      = own_canonical_form) by (vm_compute; reflexivity).
  rewrite Hrenamed, Hsame in Heq.
  assert (Hx : among 2%positive (variables_of own_canonical_form) = true)
    by (vm_compute; reflexivity).
  specialize (Hv 2%positive Hx).
  unfold own_canonical_form in Heq. cbn [rename_expr] in Heq.
  injection Heq as Hname _. rewrite Hv in Hname. vm_compute in Hname.
  discriminate Hname.
Qed.
